import os
import json
import gc
from copy import deepcopy
from pathlib import Path
from typing import Literal

import keras
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from alz_prog_net.model import AlzProgNet
from utils import compute_time_class_weights
from alz_prog_net.loss import LongitudinalTransitionLoss
from alz_prog_net.metrics import grouped_categorical_accuracy
from alz_prog_net.eval import evaluate_model


modalities = {
    "mri": (6, slice(None, 6)),
    "pet": (2, slice(6, 8)),
    "cog": (11, slice(8, 19)),
    "csf": (3, slice(19, 22)),
    "rf":  (4, slice(22, 26)),
}


def split_modalities(X, combo):
    return [
        X[:, modalities[m][1]]
        for m in combo
        if m in modalities
    ]


def cross_validation(
    X,
    y,
    y_stable,
    groups,
    combo,
    run_name,
    n_splits=5,
    random_state=42,
    imputer: Literal["mean", "median"] = "median",
    scaling: Literal["min-max", "standardization"] = "min-max",
    epochs=250,
    batch_size=32,
    time_weights=None,
    severity_matrix=None,
    severity_weight=0.5,
    transition_weight=1.0,
    transition_loss="huber",
    huber_delta=1.0,
    from_logits=False,
):
    """
    Perform grouped stratified K-fold cross-validation.

    Saves:
        models/{run_name}/model_{fold}.keras
        results/cv/{run_name}/result_{fold}.json
    """

    if time_weights is None:
        time_weights = [0.75, 1.0, 1.5, 1.25]

    if severity_matrix is None:
        severity_matrix = [
            [0.0, 0.5, 2.0],
            [0.5, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------

    model_dir = Path("models") / run_name
    results_dir = Path("results") / "cv" / run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cross-validation splitter
    # ------------------------------------------------------------------

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(X, y_stable, groups)
    ):
        print(f"\n===== Fold {fold + 1}/{n_splits} =====")

        # --------------------------------------------------------------
        # Split data
        # --------------------------------------------------------------

        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # --------------------------------------------------------------
        # Preprocessing
        #
        # IMPORTANT:
        # Fit preprocessing ONLY on the training fold.
        # --------------------------------------------------------------

        pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy=imputer
                ),
            ),
            (
                "scaler",
                (
                    MinMaxScaler()
                    if scaling == "min-max"
                    else StandardScaler()
                ),
            ),
        ])

        X_train_scaled = pipe.fit_transform(X_train_raw)
        X_val_scaled = pipe.transform(X_val_raw)

        X_train = split_modalities(X_train_scaled, combo)
        X_val = split_modalities(X_val_scaled, combo)

        # --------------------------------------------------------------
        # Fold-specific class weights
        # --------------------------------------------------------------

        class_weights = compute_time_class_weights(y_train)

        fold_model = AlzProgNet(
            num_modalities=5,
            modalities_hidden_dims=[32, 512],
            modality_output_dim=128,
            latent_dim=512,
            num_transformer_layers=3,
            num_heads=2,
            ff_dim=512,
            dropout=0.1,
            progression_hidden_dims=(264, 128, 32),
            time_points=(0, 6, 12, 24),
            temporal_levels=(True, True, False),
            time_dim=16,
            n_time_frequencies=4,
            use_time=True,
            use_gate=True,
            use_residual=True,
            initial_residual_scale=0.1,
            output_dim=3
        )

        inputs = [
            keras.Input(
                shape=(modalities[m][0],),
                name=f"modality_{i}",
            )
            for i, m in enumerate(combo)
            if m in modalities
        ]

        outputs = fold_model(inputs)

        alz_prog_net = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="AlzProgNet",
        )

        # --------------------------------------------------------------
        # Loss
        # --------------------------------------------------------------

        loss_fn = LongitudinalTransitionLoss(
            class_weights=class_weights,
            time_weights=time_weights,
            severity_matrix=severity_matrix,
            severity_weight=severity_weight,
            transition_weight=transition_weight,
            transition_loss=transition_loss,
            huber_delta=huber_delta,
            from_logits=from_logits,
        )

        optimizer = keras.optimizers.AdamW()

        alz_prog_net.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[grouped_categorical_accuracy],
        )

        # --------------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------------

        model_path = model_dir / f"model_{fold}.keras"

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.9,
                patience=10,
                min_lr=1e-6,
                verbose=0,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        # --------------------------------------------------------------
        # Training
        # --------------------------------------------------------------

        history = alz_prog_net.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )

        # --------------------------------------------------------------
        # Determine best epoch
        # --------------------------------------------------------------

        history_dict = history.history

        val_losses = history_dict["val_loss"]
        best_epoch_idx = int(min(range(len(val_losses)), key=val_losses.__getitem__))

        best_epoch = best_epoch_idx + 1
        best_val_loss = float(val_losses[best_epoch_idx])

        # Number of epochs actually executed
        epochs_trained = len(history.epoch)

        # --------------------------------------------------------------
        # Load the best checkpoint explicitly
        #
        # This makes it clear that evaluation is performed using the
        # checkpoint that achieved the best validation loss.
        # --------------------------------------------------------------

        if model_path.exists():
            alz_prog_net = keras.models.load_model(
                model_path,
                custom_objects={
                    "LongitudinalTransitionLoss": LongitudinalTransitionLoss,
                    "grouped_categorical_accuracy": grouped_categorical_accuracy,
                },
            )

        # --------------------------------------------------------------
        # Prediction
        # --------------------------------------------------------------

        y_pred = alz_prog_net.predict(
            X_val,
            verbose=0,
        )

        # --------------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------------

        results = evaluate_model(
            y_val,
            y_pred,
        )

        # --------------------------------------------------------------
        # Add training / CV metadata
        # --------------------------------------------------------------

        results = {
            **results,

            # Fold information
            "fold": fold,
            "n_splits": n_splits,

            # Training information
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "best_val_loss": best_val_loss,

            # Configuration
            "batch_size": batch_size,
            "max_epochs": epochs,
            "optimizer": "AdamW",
            "imputer": imputer,
            "scaling": scaling,
            "modalities": list(combo),

            # Loss configuration
            "time_weights": time_weights,
            "severity_matrix": severity_matrix,
            "severity_weight": severity_weight,
            "transition_weight": transition_weight,
            "transition_loss": transition_loss,
            "huber_delta": huber_delta,
            "from_logits": from_logits,

            # Fold information
            "n_train": len(train_idx),
            "n_val": len(val_idx),

            # Useful for reproducibility
            "random_state": random_state,
        }

        # --------------------------------------------------------------
        # Save training history
        #
        # Convert numpy values to regular Python floats for JSON.
        # --------------------------------------------------------------

        results["history"] = {
            metric: [float(v) for v in values]
            for metric, values in history_dict.items()
        }

        # --------------------------------------------------------------
        # Save class weights
        # --------------------------------------------------------------

        try:
            results["class_weights"] = class_weights.tolist()
        except AttributeError:
            results["class_weights"] = class_weights

        # --------------------------------------------------------------
        # Save result
        # --------------------------------------------------------------

        result_path = results_dir / f"result_{fold}.json"

        with open(result_path, "w") as f:
            json.dump(
                results,
                f,
                indent=2,
            )

        all_fold_results.append(results)

        print(
            f"Fold {fold + 1}: "
            f"best_epoch={best_epoch}, "
            f"epochs_trained={epochs_trained}, "
            f"best_val_loss={best_val_loss:.5f}"
        )

        #clean up
        del alz_prog_net
        del fold_model
        del loss_fn
        del optimizer
        del history
        del X_train_raw, X_val_raw
        del X_train_scaled, X_val_scaled
        del X_train, X_val
        del y_train, y_val
        del y_pred

        keras.backend.clear_session()
        gc.collect()

    return all_fold_results
