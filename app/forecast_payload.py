"""Lossless archive payload codec, compatible with original SQLite TEXT rows."""
import gzip
import json


def decode_payload_text(value):
    """Return the exact original UTF-8 JSON text, never reserialize it."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if raw.startswith(b"\x1f\x8b"):
            raw = gzip.decompress(raw)
        return raw.decode("utf-8")
    raise TypeError("Archive payload must be JSON text or bytes")


def encode_payload(text):
    """Deterministic gzip BLOB; SQLite permits it in a legacy TEXT column."""
    raw = decode_payload_text(text).encode("utf-8")
    return gzip.compress(raw, compresslevel=6, mtime=0)


def loads(value):
    return json.loads(decode_payload_text(value))
