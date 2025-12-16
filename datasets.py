from sklearn.datasets import load_wine, load_digits, make_classification

def get_dataset_wine():
    X, y = load_wine(return_X_y=True)
    name = "wine"
    return X, y, name


def get_dataset_digits():
    X, y = load_digits(return_X_y=True)
    name = "digits"
    return X, y, name


def get_dataset_synthetic_multiclass():
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=4,
        flip_y=0.05,
        class_sep=1.0,
        random_state=0,
    )
    name = "synthetic_4class"
    return X, y, name