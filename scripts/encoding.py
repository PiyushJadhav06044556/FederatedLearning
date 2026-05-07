#!/usr/bin/env python3
"""Minimal encoding utilities for SecAgg (clip, encode, decode)."""
from __future__ import annotations
import math
from typing import List


def l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def clip_vector(vec: List[float], C: float) -> List[float]:
    nrm = l2_norm(vec)
    if nrm <= C or nrm == 0:
        return list(vec)
    scale = C / nrm
    return [x * scale for x in vec]


def mod_q(x: int, q: int) -> int:
    r = x % q
    if r < 0:
        r += q
    return r


def encode_vector_to_int(vec: List[float], S: int = 2**16, q: int = 2**61 - 1) -> List[int]:
    # fixed-point
    return [mod_q(int(round(S * v)), q) for v in vec]


def center_lift(x: int, q: int) -> int:
    # center [0,q-1] to (-(q//2),(q//2)]
    if x > q // 2:
        return x - q
    return x


def decode_int_to_float(int_vec: List[int], S: int = 2**16, q: int = 2**61 - 1) -> List[float]:
    return [center_lift(v, q) / float(S) for v in int_vec]
