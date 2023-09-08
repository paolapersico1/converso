"""Module to perform the classification task."""

import logging
import os
from os import path

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from .const import (
    MODELS_DIR,
    SAMPLING,
    SAVE_MODELS,
    SLOTS,
    STOP_WORD_REMOVAL,
    USE_SAVED_MODELS,
)
from .data_preprocessing import W2VTransformer
from .models import classifiers

_LOGGER = logging.getLogger(__name__)


def load_and_predict(text, model_name):
    """Load model and predict output."""
    result = {}
    model_file_path, model_file_info = get_models_metadata(model_name, "Intent")
    model = load(model_file_path)
    result["Intent"] = model.predict([text])[0]
    for slot in SLOTS[result["Intent"]]:
        if not (slot == "State" and result.get("Response", "") == "one") and not (
            slot == "DeviceClass" and result["Domain"] != "cover"
        ):
            model_file_path, model_file_info = get_models_metadata(model_name, slot)
            model = load(model_file_path)
            if slot.startswith("Response"):
                slot = "Response"
            result[slot] = model.predict([text])[0]
    return result


def generate_best_models(x_train, y_train, x_test, y_test, label, w2v):
    """Load or generate the best models."""
    best_models = {}

    models = classifiers

    for clf_name, model, params in models:
        model_name = clf_name
        model_file_path, model_info_file_path = get_models_metadata(model_name, label)

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
            result, current_model = grid_search(x_train, y_train, model, params, w2v)
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


def grid_search(x_trainval, y_trainval, clf, params, w2v):
    """Perform grid search with different parameters."""
    if SAMPLING == "undersampling":
        rs = RandomUnderSampler()
    else:
        rs = RandomOverSampler()
    if STOP_WORD_REMOVAL:
        embedder = W2VTransformer(w2v)
    else:
        embedder = W2VTransformer(w2v, with_sw=False)

    pipeline = Pipeline(
        [
            ("sampling", rs),
            ("embedding", embedder),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )

    gs = GridSearchCV(
        pipeline,
        params,
        cv=5,
        n_jobs=1,
        return_train_score=True,
        verbose=3,
    )
    gs.fit(x_trainval, y_trainval)

    return pd.DataFrame(gs.cv_results_), gs.best_estimator_


def get_models_metadata(model_name, label):
    """Return model metadata."""
    sw_removal = ""
    if STOP_WORD_REMOVAL:
        sw_removal = "without_sw"
    model_file_name = label + "__" + model_name + "__" + SAMPLING + "__" + sw_removal
    model_file_path = path.join(MODELS_DIR, model_file_name + ".joblib")
    model_info_file_path = path.join(MODELS_DIR, model_file_name + ".csv")

    return model_file_path, model_info_file_path
