"""Model parameters."""
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

classifiers = [
    (
        "Linear_SVC",
        SVC(kernel="linear", max_iter=10000),
        {"clf__C": [0.1, 1, 10, 100]},
    ),
    (
        "Polinomial_SVC",
        SVC(kernel="poly", max_iter=10000),
        {"clf__C": [0.1, 1, 10, 100], "clf__degree": [2, 3]},
    ),
    ("Gaussian_NB", GaussianNB(), {}),
    ("KNN", KNeighborsClassifier(), {"clf__n_neighbors": range(1, 5)}),
    (
        "MLP",
        MLPClassifier(solver="sgd", max_iter=10000),
        {
            "clf__hidden_layer_sizes": [(100,), (100, 10)],
            "clf__alpha": [0.0001, 0.05],
        },
    ),
]
