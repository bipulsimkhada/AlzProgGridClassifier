from keras import ops


def grouped_categorical_accuracy(y_true, y_pred):
    """
    Computes categorical accuracy averaged across 4 groups.

    y_true: (N, 4, 3)
    y_pred: (N, 4, 3)

    Last dimension:
        0 = CN
        1 = MCI
        2 = AD

    Returns:
        Scalar accuracy averaged across all samples and groups.
    """

    # Convert one-hot labels to class indices
    # Shape: (N, 4)
    true_labels = ops.argmax(y_true, axis=-1)

    # Convert predictions to predicted class indices
    # Shape: (N, 4)
    pred_labels = ops.argmax(y_pred, axis=-1)

    # Correct/incorrect for each group
    # Shape: (N, 4)
    correct = ops.cast(
        ops.equal(true_labels, pred_labels),
        "float32"
    )

    # Average across groups and batch
    return ops.mean(correct)
