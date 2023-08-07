"""Module to perform the classification task."""

import os
from os import path

from joblib import dump, load
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classifiers import (
    classifiers,
)
from .consts import (
    MODELS_DIR,
    SAVE_MODELS,
    USE_SAVED_MODELS,
)


def generate_best_models(x_train, y_train, x_test, y_test, dataset_name):
    """Load or generate the best models."""
    best_models = {}

    for clf_name, model, params in classifiers:
        model_name = clf_name + "__" + dataset_name
        model_file_name, model_file_path, model_info_file_path = get_models_metadata(
            MODELS_DIR, model_name
        )

        best_models[model_name] = {}

        if (
            USE_SAVED_MODELS
            and os.access(model_file_path, os.R_OK)
            and os.access(model_info_file_path, os.R_OK)
        ):
            best_models[model_name]["model"] = load(model_file_path)
            result = pd.read_csv(model_info_file_path)
        else:
            # cross-validate
            result, current_model = grid_search(x_train, y_train, model, params)
            best_models[model_name]["model"] = current_model
            if SAVE_MODELS:
                # save the model
                dump(current_model, model_file_path)
                result.to_csv(model_info_file_path)

        attributes = [
            "mean_train_score",
            "mean_test_score",
            "mean_fit_time",
            "mean_score_time",
        ]
        for attribute in attributes:
            best_models[model_name][attribute] = result.loc[
                result["rank_test_score"] == 1
            ][attribute].values[0]

        best_models[model_name]["final_test_score"] = best_models[model_name][
            "model"
        ].score(x_test, y_test)

    return best_models


def grid_search(x_trainval, y_trainval, clf, params):
    """Perform grid search with different parameters."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    gs = GridSearchCV(
        pipeline, params, cv=2, n_jobs=16, return_train_score=True, verbose=3
    )
    gs.fit(x_trainval, y_trainval)

    return pd.DataFrame(gs.cv_results_), gs.best_estimator_


def get_models_metadata(models_dir, model_name):
    """Return model metadata."""
    model_file_name = model_name + ".joblib"
    model_file_path = path.join(models_dir, model_file_name)
    model_info_file_path = path.join(models_dir, model_name + ".csv")

    return model_file_name, model_file_path, model_info_file_path
