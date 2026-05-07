#!/usr/bin/env python3
"""
ECDH utilities using X25519 for pairwise shared secrets.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives import serialization


@dataclass
class KeyPair:
    sk: X25519PrivateKey
    pk: X25519PublicKey

    def serialize_public(self) -> bytes:
        return self.pk.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def generate_keypair() -> KeyPair:
    sk = X25519PrivateKey.generate()
    pk = sk.public_key()
    return KeyPair(sk=sk, pk=pk)


def shared_secret(sk: X25519PrivateKey, peer_pk_bytes: bytes) -> bytes:
    peer_pk = X25519PublicKey.from_public_bytes(peer_pk_bytes)
    return sk.exchange(peer_pk)
