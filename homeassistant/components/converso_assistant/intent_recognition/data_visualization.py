"""Module to generate the plots for the task."""
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_intent_distribution(y):
    """Plot the intent distribution as a bar plot."""
    plt.figure()
    pd.value_counts(y).plot.bar(title="Intent distribution")
    plt.xlabel("Intent")
    plt.ylabel("Sentences")
    plt.show()


def plot_confusion_matrices(models, x, y, n_cols=3):
    """Plot confusion matrices for different models."""
    n_rows = ceil(len(models) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(wspace=1, hspace=0.3)
    for i, (name, model) in enumerate(models.items()):
        if n_rows > 1:
            ax = axs[int(i / n_cols), i % n_cols]
        else:
            ax = axs[i % n_cols]
        ConfusionMatrixDisplay.from_estimator(
            model["model"], x, y, ax=ax, xticks_rotation="vertical"
        )
        ax.set_title(" ".join(name.split("_")[:-1]))
    fig.suptitle("Confusion Matrices per Model")
    plt.show()


def plot_testing_accuracy(scores_table, models_names, dataset_names):
    """Plot testing accuracies for different models and datasets."""
    data = [
        [
            scores_table[model_name + "__" + dataset_name]
            for dataset_name in dataset_names
        ]
        for model_name in models_names
    ]
    df = pd.DataFrame(data, columns=dataset_names, index=models_names)

    ax = df.plot.bar(rot=0)
    plt.xticks(range(len(models_names)), [x.replace("_", " ") for x in models_names])
    for p in ax.patches:
        ax.annotate(
            str(round(p.get_height() * 100)) + "%",
            (p.get_x() * 1.005, p.get_height() * 1.005),
        )
    plt.title("Testing accuracies per Dataset")
    plt.show()
