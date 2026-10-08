"""Host checks for the flash risks in this dashboard's existing partition layout.

Run: python tests/test_flash_validation.py --build-dir flash -v
No test opens a serial port or downloads a file.
"""

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "tools"))
import flash_firmware
import prepare_model

BUILD_DIR = None

# First 128 bytes of the actual partition image built from the original uploaded
# dashboard. It has a valid Espressif MD5 record, but no model partition.
ORIGINAL_PARTITION_PREFIX = bytes.fromhex(
    "aa50010200900000006000006e76730000000000000000000000000000000000"
    "aa50010100f00000001000007068795f696e6974000000000000000000000000"
    "aa5000000000010000008000666163746f727900000000000000000000000000"
    "ebebffffffffffffffffffffffffffff297fe080ae579ccf3d86f9cf52e8410e"
)


class FlashValidationTests(unittest.TestCase):
    def test_supplied_model_is_the_expected_official_bundle(self):
        data = (PROJECT_DIR / "flash/model.bin").read_bytes()
        entries = prepare_model.verify_model_image(data)
        self.assertIn("mn7_en", {entry["name"] for entry in entries})

    def test_one_byte_model_corruption_is_rejected(self):
        data = bytearray((PROJECT_DIR / "flash/model.bin").read_bytes())
        data[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare_model.verify_model_image(data)

    def test_original_valid_partition_table_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="weather-partition-check-") as directory:
            path = Path(directory) / "original.partitions.bin"
            path.write_bytes(ORIGINAL_PARTITION_PREFIX.ljust(3072, b"\xff"))
            with self.assertRaisesRegex(ValueError, "differs from this dashboard"):
                flash_firmware.check_partition_table(path)

    def test_standard_ota_boot_image_address_would_overlap_nvs(self):
        with tempfile.TemporaryDirectory(prefix="weather-range-check-") as directory:
            path = Path(directory) / "boot_app0.bin"
            path.write_bytes(bytes(4096))
            with self.assertRaisesRegex(ValueError, "erase existing NVS"):
                flash_firmware.check_write_ranges([(0xE000, path)])

    def test_compiled_images_pass_the_complete_file_preflight(self):
        if BUILD_DIR is None:
            self.skipTest("Pass --build-dir to check the actual compiled image set")
        bootloader, partitions, application = flash_firmware.find_build_images(BUILD_DIR)
        flash_firmware.check_partition_table(partitions)
        flash_firmware.check_esp_image(bootloader, 0x8000)
        flash_firmware.check_esp_image(application, 0x800000)
        model = PROJECT_DIR / "flash/model.bin"
        prepare_model.verify_model_image(model.read_bytes())
        flash_firmware.check_write_ranges([
            (0, bootloader), (0x8000, partitions),
            (0x10000, application), (0x810000, model),
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path)
    args, remaining = parser.parse_known_args()
    BUILD_DIR = args.build_dir.resolve() if args.build_dir else None
    unittest.main(argv=[sys.argv[0]] + remaining)
