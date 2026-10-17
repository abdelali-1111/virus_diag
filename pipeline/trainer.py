"""Entraînement : responsabilité unique = apprendre à partir des données."""
from utils.preprocessing import StandardScaler


class Trainer:
    def __init__(self, model, dataset, test_size=0.25, seed=42):
        self.model = model
        self.dataset = dataset
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.seed = seed
        self.X_test = None
        self.y_test = None

    def train(self):
        X_train, X_test, y_train, y_test = self.dataset.train_test_split(
            self.test_size, self.seed)
        X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_test = y_test
        self.model.fit(X_train, y_train)
        return self.model
