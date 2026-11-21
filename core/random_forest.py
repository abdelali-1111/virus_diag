"""Random Forest implémentée avec numpy."""

import numpy as np
from core.model import BaseModel


class _DecisionTree:
    """Arbre de décision CART interne (régression par MSE ou classification)."""

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, task="regression"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.task = task
        self.tree = None

    def fit(self, X, y, rng):
        self.tree = self._build_tree(X, y, depth=0, rng=rng)
        return self

    def _leaf_value(self, y):
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        return float(np.mean(y))

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        if self.task == "classification":
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return float(1.0 - np.sum(probs ** 2))  # Indice de Gini
        mean_y = np.mean(y)
        return float(np.mean((y - mean_y) ** 2))  # MSE

    def _best_split(self, X, y, rng):
        n, d = X.shape
        if n < self.min_samples_split:
            return None, None, None

        parent_imp = self._impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        k = self.max_features or d
        k = min(max(1, k), d)
        feat_indices = rng.choice(d, size=k, replace=False)

        for feat in feat_indices:
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                n_left = np.sum(left_mask)
                n_right = n - n_left
                if n_left == 0 or n_right == 0:
                    continue

                gain = parent_imp - (
                    (n_left / n) * self._impurity(y[left_mask])
                    + (n_right / n) * self._impurity(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = float(thresh)

        if best_gain <= 0.0:
            return None, None, None
        return best_feat, best_thresh, best_gain

    def _build_tree(self, X, y, depth, rng):
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return {"leaf": True, "value": self._leaf_value(y)}

        feat, thresh, _ = self._best_split(X, y, rng)
        if feat is None:
            return {"leaf": True, "value": self._leaf_value(y)}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1, rng),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1, rng),
        }

    def _predict_row(self, x, node):
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])


class RandomForest(BaseModel):
    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        max_features="sqrt",
        task="regression",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.task = task
        self.random_state = random_state
        self.trees = []

    def _resolve_max_features(self, d):
        if isinstance(self.max_features, int):
            return self.max_features
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * d))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(np.log2(d)))
        return d

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if self.task == "regression":
            y = y.astype(float)

        n, d = X.shape
        k = self._resolve_max_features(d)
        rng = np.random.default_rng(self.random_state)
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap (tirage avec remise)
            indices = rng.integers(0, n, size=n)
            X_boot, y_boot = X[indices], y[indices]

            tree = _DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=k,
                task=self.task,
            )
            tree.fit(X_boot, y_boot, rng)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        all_preds = np.array([tree.predict(X) for tree in self.trees])  # (n_estimators, n_samples)

        if self.task == "classification":
            preds = []
            for col in range(X.shape[0]):
                values, counts = np.unique(all_preds[:, col], return_counts=True)
                preds.append(values[np.argmax(counts)])
            return np.array(preds)

        return np.mean(all_preds, axis=0)
