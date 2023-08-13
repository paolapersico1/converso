"""Module to perform the classification task."""

import logging
import os
from os import path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import skops.io as sio

from .const import (
    MODELS_DIR,
    SAVE_MODELS,
    SLOTS,
    USE_SAVED_MODELS,
)
from .models import classifiers, regressors

_LOGGER = logging.getLogger(__name__)


def load_and_predict(X, model_name):
    """Load model and predict output."""
    result = {}
    model_file_path, model_file_info = get_models_metadata(model_name, "Intent")
    model = sio.load(model_file_path, trusted=True)
    result["Intent"] = model.predict(X)[0]
    for slot in SLOTS[result["Intent"]]:
        if not (slot == "Response" and result.get("State", "") == "one") and not (
            slot == "DeviceClass" and result["Domain"] != "cover"
        ):
            model_file_path, model_file_info = get_models_metadata(model_name, slot)
            model = sio.load(model_file_path, trusted=True)
            result[slot] = model.predict(X)[0]
    return result


def generate_best_models(x_train, y_train, x_test, y_test, dataset_name, label):
    """Load or generate the best models."""
    best_models = {}

    models = []
    if label not in ("Brightness", "Temperature"):
        models = classifiers
    else:
        models = regressors

    for clf_name, model, params in models:
        model_name = clf_name + "__" + dataset_name
        model_file_path, model_info_file_path = get_models_metadata(model_name, label)

        best_models[model_name] = {}

        if (
            USE_SAVED_MODELS
            and os.access(model_file_path, os.R_OK)
            and os.access(model_info_file_path, os.R_OK)
        ):
            best_models[model_name]["model"] = sio.load(model_file_path, trusted=True)
            result = pd.read_csv(model_info_file_path)
        else:
            # cross-validate
            result, current_model = grid_search(x_train, y_train, model, params)
            best_models[model_name]["model"] = current_model
            if SAVE_MODELS:
                # save the model
                sio.dump(current_model, model_file_path)
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
        pipeline, params, cv=4, n_jobs=4, return_train_score=True, verbose=3
    )
    gs.fit(x_trainval, y_trainval)

    return pd.DataFrame(gs.cv_results_), gs.best_estimator_


def get_models_metadata(model_name, label):
    """Return model metadata."""
    model_file_name = label + "__" + model_name
    model_file_path = path.join(MODELS_DIR, model_file_name + ".skops")
    model_info_file_path = path.join(MODELS_DIR, model_file_name + ".csv")

    return model_file_path, model_info_file_path
