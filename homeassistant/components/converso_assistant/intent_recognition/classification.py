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
    USE_SAVED_MODELS,
    WORD2VEC_DIM,
)
from .data_preprocessing import W2VTransformer
from .models import classifiers

_LOGGER = logging.getLogger(__name__)


def load_and_predict(text, w2v):
    """Load model and predict output."""
    result = {}

    embedder_without_sw = W2VTransformer(w2v, with_sw=False)
    embedder_with_sw = W2VTransformer(w2v, with_sw=True)
    embedding_without_sw = embedder_without_sw.transform([text])
    embedding_with_sw = embedder_with_sw.transform([text])

    intent_model = load(
        path.join(
            MODELS_DIR,
            "best_models",
            "Intent.joblib",
        ),
    )
    try:
        result["Intent"] = intent_model.predict(embedding_without_sw)[0]
        for slot in SLOTS[result["Intent"]]:
            if not (slot == "State" and result.get("Response", "") == "one") and not (
                slot == "DeviceClass" and result["Domain"] != "cover"
            ):
                model = load(
                    path.join(
                        MODELS_DIR,
                        "best_models",
                        slot + ".joblib",
                    ),
                )
                if slot.startswith("Response"):
                    slot = "Response"
                if slot == "ResponseHassGetState":
                    result[slot] = model.predict(embedding_with_sw)[0]
                else:
                    result[slot] = model.predict(embedding_without_sw)[0]
    except ValueError:
        return None
    return result


def generate_best_models(x_train, y_train, x_test, y_test, label, w2v, without_sw):
    """Load or generate the best models."""
    best_models = {}

    models = classifiers

    for clf_name, model, params in models:
        model_name = clf_name
        if without_sw:
            model_name = model_name + "__without_sw"
        model_file_path, model_info_file_path = get_models_metadata(
            clf_name, label, without_sw
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
            result, current_model = grid_search(
                x_train, y_train, model, params, w2v, without_sw
            )
            best_models[model_name]["model"] = current_model
            if SAVE_MODELS:
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


def grid_search(x_trainval, y_trainval, clf, params, w2v, without_sw):
    """Perform grid search with different parameters."""
    if SAMPLING == "undersampling":
        rs = RandomUnderSampler(random_state=42)
    else:
        rs = RandomOverSampler(random_state=42)

    embedder = W2VTransformer(w2v, with_sw=(not without_sw))

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


def get_models_metadata(model_name, label, without_sw):
    """Return model metadata."""
    model_file_name = (
        label + "__" + model_name + "__" + SAMPLING + "__" + str(WORD2VEC_DIM)
    )
    if without_sw:
        model_file_name = model_file_name + "__without_sw"
    model_file_path = path.join(MODELS_DIR, model_file_name + ".joblib")
    model_info_file_path = path.join(MODELS_DIR, model_file_name + ".csv")

    return model_file_path, model_info_file_path
