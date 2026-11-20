import math
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm


def suggest_parameters(trial, classifier_techniques, classifier_name):

    params_distribution = classifier_techniques[
        classifier_name
    ]["params_distribution"]

    params = {}

    for param_name, config in params_distribution.items():

        if config["type"] == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                config["low"],
                config["high"],
                log=config.get("log", False)
            )

        elif config["type"] == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name,
                config["choices"]
            )

    return params


def fit_scaler(train_iterator):

    scaler = StandardScaler()

    total_batches = math.ceil(
        train_iterator.sample_size / train_iterator.batch_size
    )

    for X_batch, y_batch in tqdm(
        train_iterator.get_batches(),
        total=total_batches,
        desc="Fitting scaler",
        leave=True
    ):
        scaler.partial_fit(X_batch)

    return scaler


def train_model(model, train_iterator, scaler, epoch):

    classes = np.array([0, 1])

    total_batches = math.ceil(
        train_iterator.sample_size / train_iterator.batch_size
    )

    for X_batch, y_batch in tqdm(
        train_iterator.get_batches(),
        total=total_batches,
        desc=f"Training epoch {epoch + 1}",
        leave=True
    ):

        if scaler is not None:
            X_batch = scaler.transform(X_batch)

        model.partial_fit(
            X_batch,
            y_batch,
            classes=classes
        )

    return model


def evaluate_model(model, validation_iterator, scaler=None):

    y_true = []
    y_scores = []

    total_batches = math.ceil(
        validation_iterator.sample_size /
        validation_iterator.batch_size
    )

    for X_batch, y_batch in tqdm(
        validation_iterator.get_batches(),
        total=total_batches,
        desc="Evaluating",
        leave=True
    ):

        if scaler is not None:
            X_batch = scaler.transform(X_batch)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_batch)[:, 1]

        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_batch)

        else:
            raise ValueError(
                f"{type(model).__name__} does not support "
                "predict_proba() or decision_function()."
            )

        y_true.extend(y_batch)
        y_scores.extend(scores)

    return average_precision_score(y_true, y_scores)