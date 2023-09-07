"""Module to generate the best model for Intent Recognition from scratch."""
import logging
from os import makedirs, path

from intent_recognition.classification import generate_best_models
from intent_recognition.const import DATASETS_DIR, MODELS_DIR, WORD2VEC_DIM
from intent_recognition.data_acquisition import load_synthetic_dataset
from intent_recognition.data_preprocessing import create_label_datasets
from intent_recognition.data_visualization import (
    plot_confusion_matrices,
    plot_distribution,
    plot_testing_accuracy,
)
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from word2vec.word2vec import Word2Vec

_LOGGER = logging.getLogger(__name__)


def set_display_options():
    """Set pandas display options."""
    pd.options.display.width = None
    pd.options.display.max_columns = None
    pd.set_option("display.max_rows", 3000)
    pd.set_option("display.max_columns", 3000)
    tqdm.pandas()


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
    table["Test accuracy"] = [f"{x:.2f}" for x in models.loc["final_test_score"]]

    table.set_index("Model", inplace=True)
    table.sort_values(by=["Test accuracy"], inplace=True, ascending=False)
    _LOGGER.info(table)


def pipeline():
    """Generate Intent Recognition models."""
    set_display_options()
    # nltk.download("stopwords")

    show_plots = False

    if not path.exists(MODELS_DIR):
        makedirs(MODELS_DIR)
    if not path.exists(DATASETS_DIR):
        makedirs(DATASETS_DIR)

    w2v = Word2Vec(dim=WORD2VEC_DIM)

    df = load_synthetic_dataset()

    for label in (
        "ResponseHassTurn",
        "ResponseHassGetState",
        "ResponseHassLightSet",
        "Intent",
        "Domain",
        "DeviceClass",
        "State",
    ):
        if show_plots:
            plot_distribution(df[label], label)

        label_dataset = create_label_datasets(
            label,
            df,
        )

        best_models = {}

        _LOGGER.info("\nLabel: " + label + " - Training\n")

        x_current = label_dataset["Text"].to_numpy().reshape(-1, 1)
        if label.startswith("Response"):
            y_current = label_dataset["Response"]
        else:
            y_current = label_dataset[label]

        x_train, x_test, y_train, y_test = train_test_split(
            x_current,
            y_current,
            test_size=0.10,
            random_state=42,
            stratify=y_current,
        )

        current_bests = generate_best_models(
            x_train, y_train, x_test, y_test, label, w2v
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
                pd_models.transpose()["final_test_score"], models_names, label_dataset
            )
