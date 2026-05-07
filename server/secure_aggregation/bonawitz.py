#!/usr/bin/env python3
"""Bonawitz-style mask helpers"""
from __future__ import annotations
from typing import Dict, List, Tuple

from .ecdh import KeyPair, generate_keypair, shared_secret
from .hkdf_prg import hkdf_expand, prg_mask_vector


def derive_pairwise_seed(my_sk, peer_pub_bytes: bytes, round_id: int, i: str, j: str) -> bytes:
    ss = shared_secret(my_sk, peer_pub_bytes)
    info = f"round:{round_id}|i:{i}|j:{j}".encode()
    salt = b"secagg-hkdf"
    return hkdf_expand(ss, salt=salt, info=info, length=32)


def make_mask_vector(my_kp: KeyPair, peers: Dict[str, bytes], round_id: int, dim: int, q: int, my_id: str) -> List[int]:
    """Sum signed pairwise masks; signs cancel across clients."""
    acc = [0] * dim
    for pid, ppub in peers.items():
        if pid == my_id:
            continue
        if my_id < pid:
            seed = derive_pairwise_seed(my_kp.sk, ppub, round_id, my_id, pid)
            r = prg_mask_vector(seed, dim, q)
            for k in range(dim):
                acc[k] = (acc[k] + r[k]) % q
        else:
            seed = derive_pairwise_seed(my_kp.sk, ppub, round_id, pid, my_id)
            r = prg_mask_vector(seed, dim, q)
            for k in range(dim):
                acc[k] = (acc[k] - r[k]) % q
    return acc
