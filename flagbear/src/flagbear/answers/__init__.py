#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: __init__.py
#   Author: xyy15926
#   Created: 2026-07-03 10:00:00
#   Updated: 2026-07-03 10:00:00
#   Description: Answers subpackage - bit manipulation, scalar, random utilities
# ---------------------------------------------------------

from flagbear.answers.ans_bits import (
    count_one,
    count_one_odd,
    reverse_bits,
    sqrt,
)
from flagbear.answers.ans_scalar import (
    euclid_gcd,
    euclid_lcm,
    poly_eval,
)
from flagbear.answers.ans_random import (
    box_muller,
    metropolis,
    metropolis_hastings,
)

__all__ = [
    "count_one",
    "count_one_odd",
    "reverse_bits",
    "sqrt",
    "euclid_gcd",
    "euclid_lcm",
    "poly_eval",
    "box_muller",
    "metropolis",
    "metropolis_hastings",
]
