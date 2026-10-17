"""Chargement et découpage du jeu de données patients."""
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, csv_path, target="infected"):
        self.df = pd.read_csv(csv_path)
        self.target = target
        self.feature_names = [c for c in self.df.columns if c != target]

    @property
    def X(self):
        return self.df[self.feature_names].to_numpy(dtype=float)

    @property
    def y(self):
        return self.df[self.target].to_numpy(dtype=float)

    def train_test_split(self, test_size=0.25, seed=42):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.df))
        n_test = int(len(idx) * test_size)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        return (self.X[train_idx], self.X[test_idx],
                self.y[train_idx], self.y[test_idx])

    def __len__(self):
        return len(self.df)
