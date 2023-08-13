"""Model parameters."""
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

classifiers = [
    (
        "svc_linear",
        SVC(kernel="linear", max_iter=100000),
        {"clf__C": np.logspace(-1, 1, 3)},
    ),
    (
        "svc_poly",
        SVC(kernel="poly", max_iter=100000),
        {"clf__C": np.logspace(-1, 1, 3), "clf__degree": [2, 3]},
    ),
    ("Gaussian_NB", GaussianNB(), {}),
    ("KNN", KNeighborsClassifier(), {"clf__n_neighbors": range(1, 5)}),
]

regressors = [
    (
        "svr_linear",
        SVR(kernel="linear", max_iter=100000),
        {"clf__C": np.logspace(-1, 1, 3)},
    ),
    (
        "svr_poly",
        SVR(kernel="poly", max_iter=100000),
        {"clf__C": np.logspace(-1, 1, 3), "clf__degree": [2, 3]},
    ),
    (
        "sgd",
        SGDRegressor(),
        {
            "clf__alpha": 10.0 ** -np.arange(1, 3),
            "clf__learning_rate": ["constant", "optimal", "invscaling"],
        },
    ),
]
