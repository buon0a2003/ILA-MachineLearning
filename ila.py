# ila.py
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple
from data import EncodedData


def generate_combinations(n: int, k: int):
    if k < 1 or k > n:
        return []
    return list(combinations(range(n), k))


def ILA(encoded: EncodedData) -> List[List[int]]:
    X, y = encoded.X, encoded.y
    n_attr = len(encoded.headers)

    sub_idx: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(y):
        sub_idx[label].append(i)

    def key_of(row_idx: int, comb: Tuple[int, ...]) -> str:
        return "|".join(str(X[row_idx][a]) for a in comb)

    rules: List[List[int]] = []
    for label, idxs in sub_idx.items():
        mask = [False] * len(idxs)
        covered = 0
        j = 1
        while covered < len(idxs):
            if j > n_attr:
                break  # tránh loop vô hạn
            best_comb: Tuple[int, ...] = tuple()
            best_rows_local: List[int] = []
            max_count = 0

            for comb in generate_combinations(n_attr, j):
                # nhóm các dòng (chưa cover) của lớp hiện tại theo khóa
                freq: Dict[str, List[int]] = defaultdict(list)
                for local_i, row_idx in enumerate(idxs):
                    if mask[local_i]:
                        continue
                    freq[key_of(row_idx, comb)].append(local_i)

                if not freq:
                    continue

                # xây set khóa của các lớp khác
                other_keys = set()
                for other_label, other_idxs in sub_idx.items():
                    if other_label == label:
                        continue
                    for row_idx in other_idxs:
                        other_keys.add(key_of(row_idx, comb))

                # chọn nhóm khóa "duy nhất" có nhiều dòng nhất
                for k_str, local_ids in freq.items():
                    if k_str not in other_keys and len(local_ids) > max_count:
                        max_count = len(local_ids)
                        best_comb = comb
                        best_rows_local = local_ids

            if max_count > 0:
                rule = [-1] * (n_attr + 1)  # +1 cho class
                first_row_idx = idxs[best_rows_local[0]]
                for a in best_comb:
                    rule[a] = X[first_row_idx][a]
                rule[-1] = label
                rules.append(rule)

                for local_i in best_rows_local:
                    if not mask[local_i]:
                        mask[local_i] = True
                        covered += 1
            else:
                j += 1

    return rules
