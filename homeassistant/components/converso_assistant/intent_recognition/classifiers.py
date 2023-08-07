"""Classifier parameters."""
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

classifiers = [
    (
        "svc_linear",
        SVC(kernel="linear", probability=True),
        {"clf__C": [0.1, 1, 10, 100]},
    ),
    (
        "svc_poly",
        SVC(kernel="poly", probability=True),
        {"clf__C": [0.1, 1, 10, 100], "clf__degree": [2, 3]},
    ),
    (
        "svc_rbf",
        SVC(kernel="rbf", probability=True),
        {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": np.logspace(-2, 2, 3, dtype=np.float32),
        },
    ),
    ("Gaussian_NB", GaussianNB(), {}),
    ("KNN", KNeighborsClassifier(), {"clf__n_neighbors": range(1, 5)}),
]
