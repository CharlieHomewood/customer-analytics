def train_sklearn_in_batches(
    pipeline,
    data_iterator,
    classes=np.array([0, 1])
):

    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    # First pass: calculate scaling statistics
    for X_batch, y_batch in data_iterator.get_batches():

        scaler.partial_fit(X_batch)

    # Second pass: train the classifier
    for batch_number, (X_batch, y_batch) in enumerate(
        data_iterator.get_batches(),
        start=1
    ):

        X_scaled = scaler.transform(X_batch)

        classifier.partial_fit(
            X_scaled,
            y_batch,
            classes=classes
            if batch_number == 1
            else None
        )

    return pipeline