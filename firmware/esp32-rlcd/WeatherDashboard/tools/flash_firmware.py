#!/usr/bin/env python3
"""Flash this dashboard's factory layout while preserving its existing NVS.

Only bootloader, partition table, application, and speech-model ranges are written.
The original nvs (0x9000..0xEFFF) and phy_init (0xF000..0xFFFF) remain untouched.
"""

import argparse
import hashlib
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

from prepare_model import (
    MODEL_BYTES,
    MODEL_OFFSET,
    MODEL_PARTITION_BYTES,
    MODEL_SHA256,
    PROJECT_DIR,
    verify_model_image,
)

ESPTOOL_VERSION = "5.3.1"
FLASH_BYTES = 0x1000000
SECTOR_BYTES = 0x1000
EXPECTED_PARTITIONS = {
    "nvs": (1, 2, 0x9000, 0x6000, 0),
    "phy_init": (1, 1, 0xF000, 0x1000, 0),
    "factory": (0, 0, 0x10000, 0x800000, 0),
    "model": (1, 0x82, MODEL_OFFSET, MODEL_PARTITION_BYTES, 0),
}


def check_partition_table(path):
    data = path.read_bytes()
    if not data or len(data) > SECTOR_BYTES or len(data) % 32:
        raise ValueError("Invalid partition-table image length")
    partitions = {}
    md5_seen = False
    for cursor in range(0, len(data), 32):
        record = data[cursor:cursor + 32]
        if record == b"\xff" * 32:
            if any(byte != 0xFF for byte in data[cursor:]):
                raise ValueError("Unexpected data after partition-table terminator")
            break
        magic = struct.unpack_from("<H", record)[0]
        if magic == 0xEBEB:
            if md5_seen or record[2:16] != b"\xff" * 14:
                raise ValueError("Invalid partition-table checksum record")
            if hashlib.md5(data[:cursor]).digest() != record[16:32]:
                raise ValueError("Partition-table MD5 does not match its entries")
            md5_seen = True
            continue
        if magic != 0x50AA or md5_seen:
            raise ValueError("Invalid partition-table record")
        _, kind, subtype, offset, size, label, flags = struct.unpack("<HBBII16sI", record)
        name = label.split(b"\0", 1)[0].decode("ascii")
        if name in partitions:
            raise ValueError(f"Duplicate partition label: {name}")
        partitions[name] = (kind, subtype, offset, size, flags)
    if not md5_seen:
        raise ValueError("Partition-table MD5 record is missing")
    if partitions != EXPECTED_PARTITIONS:
        raise ValueError(
            "Compiled partition table differs from this dashboard's factory/NVS/model layout. "
            "Rebuild with the included partitions.csv; do not use an Arduino stock SR partition scheme."
        )


def check_esp_image(path, maximum_bytes):
    data = path.read_bytes()
    if not 24 <= len(data) <= maximum_bytes or data[0] != 0xE9:
        raise ValueError(f"{path.name}: invalid ESP image or exceeds its flash region")
    if struct.unpack_from("<H", data, 12)[0] != 9:
        raise ValueError(f"{path.name}: image was not built for ESP32-S3")
    if data[3] >> 4 != 4:
        raise ValueError(f"{path.name}: rebuild with Flash Size set to 16 MB")
    segment_count = data[1]
    if not 1 <= segment_count <= 16:
        raise ValueError(f"{path.name}: invalid ESP image segment count")
    cursor = 24
    checksum = 0xEF
    for _ in range(segment_count):
        if cursor + 8 > len(data):
            raise ValueError(f"{path.name}: truncated segment header")
        _, segment_size = struct.unpack_from("<II", data, cursor)
        cursor += 8
        if cursor + segment_size > len(data):
            raise ValueError(f"{path.name}: truncated image segment")
        for byte in data[cursor:cursor + segment_size]:
            checksum ^= byte
        cursor += segment_size
    checksum_end = (cursor + 16) & ~15
    if checksum_end > len(data) or data[checksum_end - 1] != checksum:
        raise ValueError(f"{path.name}: ESP image checksum failed")
    if data[23] != 1 or checksum_end + 32 > len(data):
        raise ValueError(f"{path.name}: ESP image SHA-256 trailer is missing")
    if hashlib.sha256(data[:checksum_end]).digest() != data[checksum_end:checksum_end + 32]:
        raise ValueError(f"{path.name}: ESP image SHA-256 failed")


def find_build_images(directory):
    candidates = []
    for partition in directory.glob("*.partitions.bin"):
        prefix = partition.name.removesuffix(".partitions.bin")
        application = directory / f"{prefix}.bin"
        bootloader = directory / f"{prefix}.bootloader.bin"
        if application.is_file() and bootloader.is_file():
            candidates.append((bootloader, partition, application))
    if len(candidates) != 1:
        raise ValueError(
            "--build-dir must contain exactly one matching .bin / .bootloader.bin / "
            ".partitions.bin set (for example WeatherDashboard.ino.*). "
            "Use Arduino Sketch > Export Compiled Binary or arduino-cli compile --build-path."
        )
    return candidates[0]


def check_write_ranges(images):
    previous_end = 0
    protected = [(0x9000, 0xF000), (0xF000, 0x10000)]
    for offset, path in sorted(images):
        size = path.stat().st_size
        end = (offset + size + SECTOR_BYTES - 1) & ~(SECTOR_BYTES - 1)
        if offset % SECTOR_BYTES or end > FLASH_BYTES or offset < previous_end:
            raise ValueError("Flash-image erase ranges overlap or exceed 16 MB")
        if any(offset < high and end > low for low, high in protected):
            raise ValueError("Flash image would erase existing NVS or PHY data")
        previous_end = end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Serial port, for example COM5 or /dev/ttyACM0; required when flashing")
    parser.add_argument("--build-dir", required=True, type=Path, help="Directory containing the three compiled sketch images")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "flash" / "model.bin")
    parser.add_argument("--esptool", type=Path, help="Optional Arduino esptool 5.3.1 executable; otherwise uses python -m esptool")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--dry-run", action="store_true", help="Validate files and show the command without opening a serial port")
    args = parser.parse_args()
    try:
        if not args.port and not args.dry_run:
            raise ValueError("Specify --port (or --dry-run to inspect without connecting)")
        if args.baud <= 0:
            raise ValueError("--baud must be positive")
        bootloader, partitions, application = find_build_images(args.build_dir.resolve())
        model = args.model.resolve()
        check_partition_table(partitions)
        check_esp_image(bootloader, 0x8000)
        check_esp_image(application, 0x800000)
        verify_model_image(model.read_bytes())
        images = [(0, bootloader), (0x8000, partitions), (0x10000, application), (MODEL_OFFSET, model)]
        check_write_ranges(images)

        if args.esptool:
            tool_path = str(args.esptool.resolve())
            command = [sys.executable, tool_path] if tool_path.endswith(".py") else [tool_path]
        else:
            command = [sys.executable, "-m", "esptool"]
        prefix = command[:]
        command += [
            "--chip", "esp32s3", "--port", args.port or "<PORT>", "--baud", str(args.baud),
            "--before", "default-reset", "--after", "hard-reset", "write-flash",
            "--flash-mode", "keep", "--flash-freq", "keep", "--flash-size", "keep",
        ]
        for offset, path in images:
            command += [f"0x{offset:x}", str(path)]

        print("Verified ESP32-S3 images, 16 MB headers, exact partition table, and model SHA-256.")
        print("Write ranges:")
        for offset, path in images:
            print(f"  0x{offset:06X}  {path.stat().st_size:>9,} bytes  {path.name}")
        print("NVS at 0x9000..0xEFFF and PHY at 0xF000..0xFFFF are excluded.")
        print("Command:")
        print(subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command))
        if args.dry_run:
            print("Dry run complete; no serial port was opened and nothing was flashed.")
            return
        version = subprocess.run(prefix + ["version"], capture_output=True, text=True, check=False)
        version_text = version.stdout + version.stderr
        reported_version = re.search(r"\besptool(?:\.py)?\s+v?([0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9.+-]*))", version_text)
        if version.returncode != 0 or not reported_version or reported_version.group(1) != ESPTOOL_VERSION:
            raise ValueError(
                "esptool 5.3.1 is required. Install it with "
                f"'{sys.executable} -m pip install esptool==5.3.1', or provide --esptool."
            )
        subprocess.run(command, check=True)
    except (OSError, ValueError, struct.error, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Flash preparation failed: {error}\n")


if __name__ == "__main__":
    main()
