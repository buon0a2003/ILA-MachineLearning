# utils.py
from __future__ import annotations
from typing import List
from data import EncodedData


def format_rule(rule: List[int], enc: EncodedData) -> str:
    n_attr = len(enc.headers)
    parts = []
    for i in range(n_attr):
        v = rule[i]
        if v != -1:
            parts.append(f"{enc.headers[i]} = {enc.inv_maps[i][v]}")
    cls = enc.inv_map_y[rule[-1]]
    cond = " AND ".join(parts) if parts else "TRUE"
    return f"IF {cond} THEN {enc.class_name} = {cls}"
