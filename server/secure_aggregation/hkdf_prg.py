#!/usr/bin/env python3
"""HKDF + HMAC-PRG: expand shared seed into mask vector mod q."""
from __future__ import annotations
import hmac
import hashlib
from typing import List

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def hkdf_expand(key_material: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    hk = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
    )
    return hk.derive(key_material)


def prg_stream(seed: bytes, out_len_bytes: int) -> bytes:
    # HMAC-DRBG style
    out = bytearray()
    counter = 1
    prev = b""
    while len(out) < out_len_bytes:
        msg = counter.to_bytes(4, "big") + prev
        block = hmac.new(seed, msg, hashlib.sha256).digest()
        out.extend(block)
        prev = block
        counter += 1
    return bytes(out[:out_len_bytes])


def prg_mask_vector(seed: bytes, dim: int, q: int) -> List[int]:
    # dim 64-bit ints mod q
    stream = prg_stream(seed, out_len_bytes=8 * dim)
    out: List[int] = []
    for i in range(dim):
        chunk = stream[8 * i: 8 * (i + 1)]
        val = int.from_bytes(chunk, "big") % q
        out.append(val)
    return out
