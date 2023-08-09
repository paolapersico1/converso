"""Classifier parameters."""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

classifiers = [
    (
        "svc_linear",
        SVC(kernel="linear", max_iter=100000),
        {"clf__C": [0.1, 1, 10]},
    ),
    (
        "svc_poly",
        SVC(kernel="poly", max_iter=100000),
        {"clf__C": [0.1, 1, 10], "clf__degree": [2, 3]},
    ),
    (
        "svc_rbf",
        SVC(kernel="rbf", max_iter=100000),
        {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": np.logspace(-2, 2, 3, dtype=np.float32),
        },
    ),
    ("Gaussian_NB", GaussianNB(), {}),
    ("KNN", KNeighborsClassifier(), {"clf__n_neighbors": range(1, 5)}),
]
