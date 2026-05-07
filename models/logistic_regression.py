from __future__ import annotations
import csv
import math
import random
from typing import List, Tuple, Iterable, Optional


def sigmoid(z: float) -> float:
    # Numerically stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def add_inplace(a: List[float], b: List[float], alpha: float = 1.0) -> None:
    for i in range(len(a)):
        a[i] += alpha * b[i]


def scale_inplace(a: List[float], alpha: float) -> None:
    for i in range(len(a)):
        a[i] *= alpha


def l2_norm_sq(a: List[float]) -> float:
    return sum(x * x for x in a)


class LogisticRegression:
    def __init__(self, n_features: int, lr: float = 0.1, reg: float = 0.0, seed: int = 42):
        self.n_features = n_features
        self.lr = lr
        self.reg = reg
        self.rng = random.Random(seed)
        # Initialize weights small and bias zero
        self.w: List[float] = [self.rng.uniform(-0.01, 0.01) for _ in range(n_features)]
        self.b: float = 0.0

    def predict_proba_row(self, x: List[float]) -> float:
        return sigmoid(dot(self.w, x) + self.b)

    def predict_row(self, x: List[float], threshold: float = 0.5) -> int:
        return 1 if self.predict_proba_row(x) >= threshold else 0

    def batch_iter(self, X: List[List[float]], y: List[int], batch_size: int, shuffle: bool = True) -> Iterable[Tuple[List[List[float]], List[int]]]:
        idx = list(range(len(X)))
        if shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(X), batch_size):
            j = idx[start:start + batch_size]
            yield [X[i] for i in j], [y[i] for i in j]

    def fit(self, X: List[List[float]], y: List[int], epochs: int = 5, batch_size: int = 64, verbose: bool = True) -> None:
        n = len(X)
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            for Xb, yb in self.batch_iter(X, y, batch_size=batch_size, shuffle=True):
                # Compute gradients
                grad_w = [0.0] * self.n_features
                grad_b = 0.0
                for xi, yi in zip(Xb, yb):
                    pi = self.predict_proba_row(xi)
                    err = pi - yi  # derivative of logloss wrt z
                    for k in range(self.n_features):
                        grad_w[k] += err * xi[k]
                    grad_b += err
                m = len(Xb)
                if m == 0:
                    continue
                # Average and add L2 reg gradient
                scale = 1.0 / m
                for k in range(self.n_features):
                    grad_w[k] = grad_w[k] * scale + self.reg * self.w[k]
                grad_b = grad_b * scale
                # SGD update
                for k in range(self.n_features):
                    self.w[k] -= self.lr * grad_w[k]
                self.b -= self.lr * grad_b
                # Mini-batch loss (for logging only)
                batch_loss = 0.0
                for xi, yi in zip(Xb, yb):
                    z = dot(self.w, xi) + self.b
                    # logloss
                    if yi == 1:
                        batch_loss += -math.log(max(sigmoid(z), 1e-12))
                    else:
                        batch_loss += -math.log(max(1 - sigmoid(z), 1e-12))
                total_loss += batch_loss
            if verbose:
                avg_loss = total_loss / max(n, 1)
                if self.reg > 0:
                    avg_loss += 0.5 * self.reg * l2_norm_sq(self.w) / max(n, 1)
                print(f"Epoch {ep}/{epochs} - avg_loss={avg_loss:.6f}")

    def predict(self, X: List[List[float]], threshold: float = 0.5) -> List[int]:
        return [self.predict_row(x, threshold) for x in X]

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [self.predict_proba_row(x) for x in X]

    def get_params(self) -> Tuple[List[float], float]:
        return list(self.w), float(self.b)

    def set_params(self, w: List[float], b: float) -> None:
        if len(w) != self.n_features:
            raise ValueError("Parameter length mismatch")
        self.w = list(w)
        self.b = float(b)

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bias", self.b])
            writer.writerow(["weights"] + self.w)

    @staticmethod
    def load_csv(path: str) -> "LogisticRegression":
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            first = next(reader)
            if first[0] != "bias":
                raise ValueError("Invalid model file")
            b = float(first[1])
            second = next(reader)
            if second[0] != "weights":
                raise ValueError("Invalid model file")
            w = [float(v) for v in second[1:]]
        m = LogisticRegression(n_features=len(w))
        m.set_params(w, b)
        return m


def load_csv_xy(path: str, label_col: Optional[str] = None) -> Tuple[List[List[float]], List[int], List[str]]:
    X: List[List[float]] = []
    y: List[int] = []
    header: List[str] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Determine label index
        if label_col is None:
            label_idx = len(header) - 1
        else:
            if label_col not in header:
                raise ValueError(f"Label column '{label_col}' not in CSV header")
            label_idx = header.index(label_col)
        feat_idx = [i for i in range(len(header)) if i != label_idx]
        for row in reader:
            if not row:
                continue
            feats = [float(row[i]) for i in feat_idx]
            label = int(float(row[label_idx]))
            X.append(feats)
            y.append(label)
    return X, y, [header[i] for i in feat_idx]


if __name__ == "__main__":
    # Simple sanity check on small synthetic data
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    model = LogisticRegression(n_features=1, lr=0.5, reg=0.0, seed=1)
    model.fit(X, y, epochs=50, batch_size=2, verbose=False)
    print("Probas:", model.predict_proba(X))
