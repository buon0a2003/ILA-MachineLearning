# data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import pandas as pd


@dataclass
class EncodedData:
    ids: List[str]                  # định danh từng dòng (từ cột đầu)
    X: List[List[int]]              # ma trận thuộc tính (không gồm class)
    y: List[int]                    # cột class đã mã-hoá
    headers: List[str]              # tên cột thuộc tính 0..n_attr-1
    class_name: str                 # tên cột class
    inv_maps: List[Dict[int, Any]]  # map ngược cho từng thuộc tính
    inv_map_y: Dict[int, Any]       # map ngược cho class


def encode_series(s: pd.Series) -> Tuple[List[int], Dict[int, Any]]:
    s2 = s.fillna("∅").astype(str)
    uniques = {v: i for i, v in enumerate(pd.unique(s2))}
    inv = {i: v for v, i in uniques.items()}
    return [uniques[v] for v in s2], inv


def preprocessing_data(file_path: str) -> EncodedData:
    # Support both CSV and Excel files
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    assert df.shape[1] >= 3, "Cần >= 3 cột (ID + >=1 thuộc tính + 1 class)."

    ids = df.iloc[:, 0].astype(str).fillna("").tolist()
    df2 = df.iloc[:, 1:]  # bỏ cột ID

    *attr_cols, class_col = df2.columns.tolist()

    X_enc_cols: List[List[int]] = []
    inv_maps: List[Dict[int, Any]] = []
    for col in attr_cols:
        enc, inv = encode_series(df2[col])
        inv_maps.append(inv)
        X_enc_cols.append(enc)

    X = list(map(list, zip(*X_enc_cols))
             ) if X_enc_cols else [[] for _ in range(len(df2))]

    y_enc, inv_y = encode_series(df2[class_col])

    return EncodedData(
        ids=ids,
        X=X,
        y=y_enc,
        headers=attr_cols,
        class_name=class_col,
        inv_maps=inv_maps,
        inv_map_y=inv_y
    )
