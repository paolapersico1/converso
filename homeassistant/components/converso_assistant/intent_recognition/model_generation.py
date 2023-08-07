"""Module to generate the best model for Intent Recognition from scratch."""
import logging
from os import makedirs, path

from classification import generate_best_models
from consts import (
    DATASETS_DIR,
    MODELS_DIR,
)
from data_acquisition import load_synthetic_dataset
from data_preprocessing import create_datasets
from data_visualization import (
    plot_confusion_matrices,
    plot_distribution,
    plot_testing_accuracy,
)
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

_LOGGER = logging.getLogger(__name__)


def set_display_options():
    """Set pandas display options."""
    pd.options.display.width = None
    pd.options.display.max_columns = None
    pd.set_option("display.max_rows", 3000)
    pd.set_option("display.max_columns", 3000)


def print_models_table(models, current_label):
    """Log model information."""
    _LOGGER.info("\nBest models for " + current_label + ":\n")

    hyperparams = {}
    table = pd.DataFrame({"Model": models.columns})

    for hp in ("C", "gamma", "degree", "n_neighbors"):
        hyperparams[hp] = []
        for model in models.loc["model"]:
            value = model.named_steps.clf.get_params().get(hp, "n/a")
            hyperparams[hp].append(value)
        table[hp] = hyperparams[hp]

    table["Fit time (s)"] = [f"{x:.2f}" for x in models.loc["mean_fit_time"]]
    table["Score time (s)"] = [f"{x:.2f}" for x in models.loc["mean_score_time"]]
    table["Train accuracy"] = [f"{x:.2f}" for x in models.loc["mean_train_score"]]
    table["Validation accuracy"] = [f"{x:.2f}" for x in models.loc["mean_test_score"]]

    table.set_index("Model", inplace=True)
    table.sort_values(by=["Validation accuracy"], inplace=True, ascending=False)
    _LOGGER.info(table)


if __name__ == "__main__":
    set_display_options()
    nltk.download("stopwords")

    show_plots = False

    if not path.exists(MODELS_DIR):
        makedirs(MODELS_DIR)
    if not path.exists(DATASETS_DIR):
        makedirs(DATASETS_DIR)

    df = load_synthetic_dataset()

    labels = (
        "DeviceClass",
        "Domain",
        "State",
        "Name",
        "Intent",
        "Area",
        "Response",
    )

    for label in labels:
        if show_plots:
            plot_distribution(df[label], label)

        datasets = create_datasets(df, label)

        best_models = {}

        for dataset_name, dataset_current in datasets.items():
            _LOGGER.info("\nLabel: " + label + " - Training on " + dataset_name + "\n")

            x_current = dataset_current.iloc[:, -100:]
            y_current = dataset_current[label]

            x_train, x_test, y_train, y_test = train_test_split(
                x_current,
                y_current,
                test_size=0.20,
                random_state=42,
                stratify=y_current,
            )

            current_bests = generate_best_models(
                x_train, y_train, x_test, y_test, dataset_name, label
            )
            best_models.update(current_bests)

            if show_plots:
                plot_confusion_matrices(current_bests, x_test, y_test, n_cols=3)

        pd_models = pd.DataFrame(best_models)
        print_models_table(pd_models, label)

        if show_plots:
            models_names = pd.unique(
                [name.split("__")[0] for name in pd_models.columns]
            )
            plot_testing_accuracy(
                pd_models.transpose()["final_test_score"], models_names, datasets.keys()
            )
