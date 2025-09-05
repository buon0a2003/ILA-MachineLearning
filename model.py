# model.py
from __future__ import annotations
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from data import EncodedData, preprocessing_data
from ila import ILA
from utils import format_rule


class ILAModel:
    def __init__(self):
        self.rules: List[List[int]] = []
        self.training_enc: Optional[EncodedData] = None
        self.majority_class: Optional[int] = None
        self.is_fitted = False

    def fit(self, file_path: str) -> 'ILAModel':
        """Train the ILA model on the given dataset."""
        self.training_enc = preprocessing_data(file_path)
        self.rules = ILA(self.training_enc)

        # Calculate majority class for fallback
        if self.training_enc.y:
            cnt = Counter(self.training_enc.y)
            self.majority_class = cnt.most_common(1)[0][0]

        self.is_fitted = True
        return self

    def predict(self, file_path: str) -> List[Any]:
        """Predict classes for instances in the given dataset."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Load and preprocess test data
        test_enc = self._preprocess_test_data(file_path)

        predictions = []
        for row_vals in test_enc.X:
            predicted_class, _ = self._classify_row(row_vals)
            # Convert back to original class value
            original_class = test_enc.inv_map_y.get(predicted_class, "Unknown")
            predictions.append(original_class)

        return predictions

    def predict_with_accuracy(self, file_path: str) -> Tuple[List[Any], float]:
        """Predict classes and return accuracy if test data has labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Load test data
        test_enc = self._preprocess_test_data(file_path)
        predictions = []

        for row_vals in test_enc.X:
            predicted_class, _ = self._classify_row(row_vals)
            predictions.append(predicted_class)

        # Calculate accuracy if we have true labels
        accuracy = None
        if test_enc.y:
            correct = sum(1 for pred, true in zip(
                predictions, test_enc.y) if pred == true)
            accuracy = correct / len(test_enc.y)

        # Convert predictions to original class values
        original_predictions = [test_enc.inv_map_y.get(
            pred, "Unknown") for pred in predictions]

        return original_predictions, accuracy

    def _preprocess_test_data(self, file_path: str) -> EncodedData:
        """Preprocess test data using the same encoding as training data."""
        # Support both CSV and Excel files
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        assert df.shape[1] >= 3, "Cần >= 3 cột (ID + >=1 thuộc tính + 1 class)."

        ids = df.iloc[:, 0].astype(str).fillna("").tolist()
        df2 = df.iloc[:, 1:]  # bỏ cột ID

        *attr_cols, class_col = df2.columns.tolist()

        # Use training encoding mappings
        X_enc_cols: List[List[int]] = []
        for i, col in enumerate(attr_cols):
            if i < len(self.training_enc.inv_maps):
                # Use training encoding
                enc_values = []
                for val in df2[col].fillna("∅").astype(str):
                    # Find encoding for this value
                    found = False
                    for encoded_val, original_val in self.training_enc.inv_maps[i].items():
                        if original_val == val:
                            enc_values.append(encoded_val)
                            found = True
                            break
                    if not found:
                        # Unknown value - use -1 as special marker
                        enc_values.append(-1)
                X_enc_cols.append(enc_values)
            else:
                # New attribute not seen in training
                enc_values = [-1] * len(df2)
                X_enc_cols.append(enc_values)

        X = list(map(list, zip(*X_enc_cols))
                 ) if X_enc_cols else [[] for _ in range(len(df2))]

        # Encode class column if it exists
        y_enc = []
        inv_y = {}
        if class_col and self.training_enc.inv_map_y:
            for val in df2[class_col].fillna("∅").astype(str):
                found = False
                for encoded_val, original_val in self.training_enc.inv_map_y.items():
                    if original_val == val:
                        y_enc.append(encoded_val)
                        found = True
                        break
                if not found:
                    y_enc.append(-1)  # Unknown class
        else:
            y_enc = [-1] * len(df2)

        return EncodedData(
            ids=ids,
            X=X,
            y=y_enc,
            headers=attr_cols,
            class_name=class_col,
            inv_maps=self.training_enc.inv_maps,  # Use training mappings
            inv_map_y=self.training_enc.inv_map_y  # Use training mappings
        )

    def _classify_row(self, row_vals: List[int]) -> Tuple[int, Optional[int]]:
        """Classify a single row using learned rules."""
        for idx, rule in enumerate(self.rules):
            if self._row_matches_rule(row_vals, rule):
                return rule[-1], idx  # first-match
        return (self.majority_class if self.majority_class is not None else -1), None

    def _row_matches_rule(self, row_vals: List[int], rule: List[int]) -> bool:
        """Check if a row matches a rule."""
        for i, v in enumerate(rule[:-1]):
            if v == -1:
                continue
            if row_vals[i] != v:
                return False
        return True

    def get_rules(self) -> List[str]:
        """Get formatted rules as strings."""
        if not self.is_fitted:
            return []
        return [format_rule(rule, self.training_enc) for rule in self.rules]

    def save_model(self, file_path: str):
        """Save the trained model to a file."""
        import pickle
        model_data = {
            'rules': self.rules,
            'training_enc': self.training_enc,
            'majority_class': self.majority_class,
            'is_fitted': self.is_fitted
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, file_path: str):
        """Load a trained model from a file."""
        import pickle
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        self.rules = model_data['rules']
        self.training_enc = model_data['training_enc']
        self.majority_class = model_data['majority_class']
        self.is_fitted = model_data['is_fitted']
