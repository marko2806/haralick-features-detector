from joblib import dump, load


class Model:
    def __init__(self, model=None):
        self.model = model

    def train_model(self, X, y):
        self.model.fit(X, y)

    def make_prediction(self, X):
        return self.model.predict(X)

    def save_model(self, filepath: str, verbose=False):
        if verbose:
            print("Saving model to " + filepath)
        dump(self.model, filepath)
        if verbose:
            print("Model saved successfully")

    def load_model(self, filepath: str, verbose=False):
        if verbose:
            print("Loading existing model from " + filepath)
        self.model = load(filepath)
