import numpy as np


CLASS_NAMES = ["CN", "MCI", "AD"]

TIME_NAMES = [
    "current",
    "6_month",
    "12_month",
    "24_month",
]


def evaluate_model(
    y_true,
    y_pred,
    class_names=None,
    time_names=None,
):
    """
    Comprehensive evaluation for longitudinal classification.

    Parameters
    ----------
    y_true : array-like
        Shape: (N, 4, 3)
        One-hot encoded true labels.

    y_pred : array-like
        Shape: (N, 4, 3)
        Predicted class probabilities.

    Returns
    -------
    dict

    Evaluation includes:

        - overall classification metrics
        - per-timepoint metrics
        - per-class metrics
        - stable subjects
        - converter subjects
        - forward converters
        - reverse converters
        - mixed converters
        - transition metrics
        - exact longitudinal sequence accuracy
    """

    class_names = (
        CLASS_NAMES
        if class_names is None
        else list(class_names)
    )

    time_names = (
        TIME_NAMES
        if time_names is None
        else list(time_names)
    )

    # =========================================================
    # Convert to NumPy
    # =========================================================

    if hasattr(y_true, "numpy"):
        y_true = y_true.numpy()

    if hasattr(y_pred, "numpy"):
        y_pred = y_pred.numpy()

    y_true = np.asarray(
        y_true,
        dtype=np.float32,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float32,
    )

    # =========================================================
    # Validation
    # =========================================================

    if y_true.ndim != 3:
        raise ValueError(
            f"y_true must have shape (N, T, C), "
            f"got {y_true.shape}"
        )

    if y_pred.ndim != 3:
        raise ValueError(
            f"y_pred must have shape (N, T, C), "
            f"got {y_pred.shape}"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes. "
            f"Got {y_true.shape} and {y_pred.shape}."
        )

    n_samples, n_times, n_classes = y_true.shape

    if n_classes != len(class_names):
        raise ValueError(
            f"Expected {len(class_names)} classes, "
            f"got {n_classes}."
        )

    if n_times != len(time_names):
        raise ValueError(
            f"Expected {len(time_names)} timepoints, "
            f"got {n_times}."
        )

    # =========================================================
    # Labels
    #
    # (N, 4)
    # =========================================================

    true_labels = np.argmax(
        y_true,
        axis=-1,
    )

    pred_labels = np.argmax(
        y_pred,
        axis=-1,
    )

    # =========================================================
    # Subject-level groups
    # =========================================================

    # Stable:
    # same diagnosis at every timepoint.

    stable_mask = np.all(
        true_labels
        == true_labels[:, :1],
        axis=1,
    )

    # Converter:
    # at least one true diagnosis changes.

    converter_mask = ~stable_mask

    # ---------------------------------------------------------
    # Transition direction
    #
    # +1 / +2 = disease-state increase
    # -1 / -2 = reverse transition
    #  0      = stable transition
    # ---------------------------------------------------------

    true_delta = np.diff(
        true_labels,
        axis=1,
    )

    pred_delta = np.diff(
        pred_labels,
        axis=1,
    )

    has_forward = np.any(
        true_delta > 0,
        axis=1,
    )

    has_reverse = np.any(
        true_delta < 0,
        axis=1,
    )

    # Only forward/no-change transitions.

    forward_converter_mask = (
        converter_mask
        & has_forward
        & ~has_reverse
    )

    # Only reverse/no-change transitions.

    reverse_converter_mask = (
        converter_mask
        & has_reverse
        & ~has_forward
    )

    # At least one forward AND one reverse transition.

    mixed_converter_mask = (
        converter_mask
        & has_forward
        & has_reverse
    )

    # =========================================================
    # Helper: safe division
    # =========================================================

    def safe_divide(a, b):
        if b == 0:
            return np.nan
        return float(a / b)

    # =========================================================
    # Confusion matrix
    # =========================================================

    def confusion_matrix(
        true,
        pred,
    ):
        matrix = np.zeros(
            (n_classes, n_classes),
            dtype=np.int64,
        )

        for t, p in zip(
            true.ravel(),
            pred.ravel(),
        ):
            matrix[t, p] += 1

        return matrix

    # =========================================================
    # Classification metrics
    # =========================================================

    def classification_metrics(
        true,
        pred,
    ):
        """
        true/pred can have any shape.

        They are flattened and treated as individual
        classification decisions.
        """

        true = np.asarray(true).reshape(-1)
        pred = np.asarray(pred).reshape(-1)

        if true.size == 0:
            return empty_classification_metrics()

        cm = confusion_matrix(
            true,
            pred,
        )

        total = cm.sum()

        accuracy = safe_divide(
            np.trace(cm),
            total,
        )

        per_class = {}

        recalls = []
        precisions = []
        f1_scores = []

        for c, name in enumerate(class_names):

            tp = cm[c, c]

            fn = (
                cm[c, :].sum()
                - tp
            )

            fp = (
                cm[:, c].sum()
                - tp
            )

            support = (
                tp + fn
            )

            predicted_count = (
                tp + fp
            )

            # ---------------------------------------------
            # Recall
            # ---------------------------------------------

            recall = safe_divide(
                tp,
                support,
            )

            # ---------------------------------------------
            # Precision
            #
            # If class exists in truth but model never
            # predicts it, precision = 0.
            # ---------------------------------------------

            if predicted_count == 0:

                precision = (
                    0.0
                    if support > 0
                    else np.nan
                )

            else:

                precision = float(
                    tp / predicted_count
                )

            # ---------------------------------------------
            # F1
            # ---------------------------------------------

            if support == 0:

                f1 = np.nan

            elif (
                precision + recall
                == 0
            ):

                f1 = 0.0

            else:

                f1 = float(
                    2
                    * precision
                    * recall
                    / (
                        precision
                        + recall
                    )
                )

            per_class[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
                "predicted_count": int(
                    predicted_count
                ),
            }

            # Only include classes actually present
            # in the true labels.

            if support > 0:

                recalls.append(recall)
                precisions.append(precision)
                f1_scores.append(f1)

        balanced_accuracy = (
            float(np.mean(recalls))
            if recalls
            else np.nan
        )

        macro_precision = (
            float(np.mean(precisions))
            if precisions
            else np.nan
        )

        macro_recall = (
            float(np.mean(recalls))
            if recalls
            else np.nan
        )

        macro_f1 = (
            float(np.mean(f1_scores))
            if f1_scores
            else np.nan
        )

        return {
            "count": int(true.size),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        }

    def empty_classification_metrics():

        return {
            "count": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "confusion_matrix": (
                np.zeros(
                    (
                        n_classes,
                        n_classes,
                    ),
                    dtype=np.int64,
                ).tolist()
            ),
            "per_class": {
                name: {
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "support": 0,
                    "predicted_count": 0,
                }
                for name in class_names
            },
        }

    # =========================================================
    # Transition metrics
    # =========================================================

    def transition_metrics(
        subject_mask,
    ):
        """
        Evaluate predicted longitudinal changes.

        true_delta:
            CN -> MCI = +1
            CN -> AD  = +2
            MCI -> CN = -1
            AD -> CN  = -2
            stable    = 0
        """

        t_delta = true_delta[
            subject_mask
        ]

        p_delta = pred_delta[
            subject_mask
        ]

        if t_delta.size == 0:

            return {
                "transition_count": 0,
                "exact_delta_accuracy": np.nan,
                "direction_accuracy": np.nan,
                "transition_mae": np.nan,
                "changed_transition_accuracy": np.nan,
                "changed_transition_direction_accuracy": np.nan,
            }

        # -----------------------------------------------------
        # Exact magnitude and direction.
        #
        # e.g.
        #
        # true +2, predicted +1 -> incorrect
        # true -1, predicted -1 -> correct
        # -----------------------------------------------------

        exact_delta_accuracy = float(
            np.mean(
                t_delta
                == p_delta
            )
        )

        # -----------------------------------------------------
        # Direction only:
        #
        # -1 = reverse
        #  0 = no transition
        # +1 = forward
        # -----------------------------------------------------

        true_direction = np.sign(
            t_delta
        )

        pred_direction = np.sign(
            p_delta
        )

        direction_accuracy = float(
            np.mean(
                true_direction
                == pred_direction
            )
        )

        # -----------------------------------------------------
        # Transition magnitude error
        # -----------------------------------------------------

        transition_mae = float(
            np.mean(
                np.abs(
                    t_delta
                    - p_delta
                )
            )
        )

        # -----------------------------------------------------
        # Evaluate only actual changing transitions
        # -----------------------------------------------------

        changed = (
            t_delta != 0
        )

        if np.any(changed):

            changed_transition_accuracy = float(
                np.mean(
                    t_delta[changed]
                    == p_delta[changed]
                )
            )

            changed_transition_direction_accuracy = float(
                np.mean(
                    true_direction[changed]
                    == pred_direction[changed]
                )
            )

        else:

            changed_transition_accuracy = np.nan
            changed_transition_direction_accuracy = np.nan

        return {
            "transition_count": int(
                t_delta.size
            ),

            "changed_transition_count": int(
                changed.sum()
            ),

            "exact_delta_accuracy":
                exact_delta_accuracy,

            "direction_accuracy":
                direction_accuracy,

            "transition_mae":
                transition_mae,

            "changed_transition_accuracy":
                changed_transition_accuracy,

            "changed_transition_direction_accuracy":
                changed_transition_direction_accuracy,
        }

    # =========================================================
    # Per-timepoint evaluation
    # =========================================================

    def timepoint_metrics(
        subject_mask,
    ):

        results = {}

        for t, time_name in enumerate(
            time_names
        ):

            results[time_name] = (
                classification_metrics(
                    true_labels[
                        subject_mask,
                        t,
                    ],
                    pred_labels[
                        subject_mask,
                        t,
                    ],
                )
            )

        return results

    # =========================================================
    # Sequence metrics
    # =========================================================

    def sequence_metrics(
        subject_mask,
    ):

        true = true_labels[
            subject_mask
        ]

        pred = pred_labels[
            subject_mask
        ]

        if true.shape[0] == 0:

            return {
                "subject_count": 0,
                "exact_sequence_accuracy": np.nan,
                "mean_timepoints_correct": np.nan,
            }

        # All four timepoints correct.

        exact_sequence = np.all(
            true == pred,
            axis=1,
        )

        # Fraction of four timepoints correct
        # for each subject.

        per_subject_accuracy = np.mean(
            true == pred,
            axis=1,
        )

        return {
            "subject_count": int(
                true.shape[0]
            ),

            "exact_sequence_accuracy": float(
                np.mean(
                    exact_sequence
                )
            ),

            "mean_timepoints_correct": float(
                np.mean(
                    per_subject_accuracy
                )
            ),
        }

    # =========================================================
    # Complete cohort evaluation
    # =========================================================

    def evaluate_cohort(
        mask,
    ):

        mask = np.asarray(
            mask,
            dtype=bool,
        )

        count = int(
            np.sum(mask)
        )

        return {
            "subject_count": count,

            # Flatten all timepoints.
            "overall": classification_metrics(
                true_labels[mask],
                pred_labels[mask],
            ),

            # Separate evaluation at each horizon.
            "by_time": timepoint_metrics(
                mask
            ),

            # Longitudinal sequence correctness.
            "sequence": sequence_metrics(
                mask
            ),

            # Transition correctness.
            "transitions": transition_metrics(
                mask
            ),
        }

    # =========================================================
    # Overall per-class counts by time
    # =========================================================

    class_distribution = {}

    for t, time_name in enumerate(
        time_names
    ):

        class_distribution[
            time_name
        ] = {}

        for c, class_name in enumerate(
            class_names
        ):

            class_distribution[
                time_name
            ][class_name] = int(
                np.sum(
                    true_labels[:, t]
                    == c
                )
            )

    # =========================================================
    # Return
    # =========================================================

    return {
        "sample_count": int(
            n_samples
        ),

        "class_names": class_names,
        "time_names": time_names,

        # Dataset-level class distribution.
        "class_distribution": (
            class_distribution
        ),

        # Everybody.
        "all": evaluate_cohort(
            np.ones(
                n_samples,
                dtype=bool,
            )
        ),

        # Never changes diagnosis.
        "stable": evaluate_cohort(
            stable_mask
        ),

        # Changes at least once.
        "converter": evaluate_cohort(
            converter_mask
        ),

        # Only moves toward higher class values.
        "forward_converter": evaluate_cohort(
            forward_converter_mask
        ),

        # Only moves toward lower class values.
        "reverse_converter": evaluate_cohort(
            reverse_converter_mask
        ),

        # Has both directions somewhere in trajectory.
        "mixed_converter": evaluate_cohort(
            mixed_converter_mask
        ),

        "cohort_counts": {
            "stable": int(
                stable_mask.sum()
            ),
            "converter": int(
                converter_mask.sum()
            ),
            "forward_converter": int(
                forward_converter_mask.sum()
            ),
            "reverse_converter": int(
                reverse_converter_mask.sum()
            ),
            "mixed_converter": int(
                mixed_converter_mask.sum()
            ),
        },
    }