"""Module to generate the plots for the task."""
from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

mpl.use("Agg")


def plot_distribution(y, label):
    """Plot the distribution as a bar plot."""
    plt.figure()
    pd.value_counts(y).plot.bar(title=label + " distribution")
    plt.xlabel(label)
    plt.ylabel("Sentences")
    plt.savefig(label + "_distribution", dpi=300)


def plot_confusion_matrices(models, x, y, label, n_cols=3, without_sw=False):
    """Plot confusion matrices for different models."""
    n_rows = ceil(len(models) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(bottom=0.18, top=0.95, wspace=1, hspace=0.3)
    for i, (name, model) in enumerate(models.items()):
        if n_rows > 1:
            ax = axs[int(i / n_cols), i % n_cols]
        else:
            ax = axs[i % n_cols]
        ConfusionMatrixDisplay.from_estimator(
            model["model"], x, y, ax=ax, xticks_rotation="vertical"
        )
        ax.set_title((name.split("__")[0]).replace("_", " ").capitalize())
        ax.set_xticklabels(
            [item.get_text().replace("Hass", "") for item in ax.get_xticklabels()]
        )
        ax.set_yticklabels(
            [item.get_text().replace("Hass", "") for item in ax.get_yticklabels()]
        )
    title = "Confusion Matrices per Model"
    filename = label + "_cm"
    if without_sw:
        title = title + " (without stopwords)"
        filename = filename + "_without_sw"
    fig.suptitle(title)
    plt.savefig(filename, dpi=300)


def group_models(series, models_names):
    """Return a dataframe with models grouped by dataset size."""
    data = [
        [series[model_name + fs] for fs in ("", "__without_sw")]
        for model_name in models_names
    ]
    return pd.DataFrame(
        data, columns=("with stop-words", "without stop-words"), index=models_names
    )


def plot_testing_accuracy(scores_table, models_names, label):
    """Plot testing accuracies for different models and datasets."""
    df = group_models(scores_table, models_names)

    ax = df.plot.bar(rot=0)
    plt.xticks(
        range(len(models_names)),
        [x.replace("_", " ") for x in models_names],
    )
    for p in ax.patches:
        ax.annotate(
            str(round(p.get_height() * 100)) + "%",
            (p.get_x() * 1.005, p.get_height() * 1.005),
            fontsize=15,
        )
    plt.title("Testing accuracies for " + label + " label")
    plt.savefig(label + "_accuracies", dpi=300)


def check(row, agent):
    """Return result of Intent Recognition."""
    if row[agent + "MatchedEntities"] == "[]":
        return "No action/response"
    if row["Intent"] in ("HassGetState", "HassClimateGetTemperature"):
        if row[agent + "Intent"] != row["Intent"]:
            return "Incorrect action"
        if row[agent + "State"] != row["State"]:
            return "Incorrect response"
        if set(row[agent + "MatchedEntities"].strip("][").split(", ")) != set(
            row["MatchedEntities"].strip("][").split(",")
        ):
            return "Incorrect response"
    else:
        for label in ("Intent", "Color", "Brightness"):
            if row[agent + label] != row[label]:
                return "Incorrect action"
        if set(row[agent + "MatchedEntities"].strip("][").split(", ")) != set(
            row["MatchedEntities"].strip("][").split(",")
        ):
            return "Incorrect action"
        if row[agent + "Response"] != row["Response"]:
            return "Correct action, incorrect response"
    return "Correct action/response"


def plot_piechart_results(df):
    """Plot a pie chart of the results for each agent and input type."""
    colors = {
        "No action/response": "lightskyblue",
        "Correct action/response": "yellowgreen",
        "Correct action, incorrect response": "lightgreen",
        "Incorrect response": "gold",
        "Incorrect action": "lightcoral",
    }

    atypes = ("default__", "converso__")
    ctypes = (
        "from_text__",
        "from_speech__",
        "from_correction__",
    )

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    i = 0
    for ctype in ctypes:
        for atype in atypes:
            df[atype + ctype + "Result"] = df.apply(
                lambda row: check(row, atype + ctype), axis=1  # noqa: B023
            )
            ax = axs[int(i / 2), i % 2]
            counts = pd.value_counts(df[atype + ctype + "Result"])
            counts.plot.pie(
                labeldistance=None,
                legend=i == 1,
                ax=ax,
                autopct="%.2f%%",
                shadow=True,
                colors=[colors[v] for v in counts],
                title=(atype.capitalize() + " (" + ctype + ")")
                .replace("__", "")
                .replace("_", " "),
            )
            if i == 1:
                ax.legend(bbox_to_anchor=(1, 1.02), loc="upper left")
            ax.axis("off")
            i = i + 1
    plt.savefig("piechart.png", dpi=300)


def plot_confusion_matrices_index(df, label):
    """Plot a confusion matrix for the label for each agent and input type."""
    atypes = ("default__", "converso__")
    ctypes = (
        "from_text__",
        "from_speech__",
        "from_correction__",
    )

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(25, 25))
    plt.subplots_adjust(hspace=0.5)
    i = 0
    for ctype in ctypes:
        for atype in atypes:
            ax = axs[int(i / 2), i % 2]
            df.fillna("None", inplace=True)
            ConfusionMatrixDisplay.from_predictions(
                df[label],
                df[atype + ctype + label],
                ax=ax,
                xticks_rotation="vertical",
                normalize="all",
            )
            ax.set_title(
                (atype.capitalize() + " (" + ctype + ")")
                .replace("__", "")
                .replace("_", " ")
            )
            ax.set_xticklabels(
                [item.get_text().replace("Hass", "") for item in ax.get_xticklabels()]
            )
            ax.set_yticklabels(
                [item.get_text().replace("Hass", "") for item in ax.get_yticklabels()]
            )
            i = i + 1

    plt.savefig("confusion_matrices.png", dpi=300)
