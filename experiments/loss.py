import sys
import os
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from dataset.dataset import create_dataset, createOutputLabels

from experiments.constants import LOSS_SEARCH_STAGES, RANDOM_STATE
from experiments.cv import cross_validation


X, y, groups = create_dataset()

target_tensors = createOutputLabels(y)
y_stable = y["stable"].to_numpy()

# -------------------------
# 80% train / 20% temp
# -------------------------
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    sgkf.split(X, y_stable, groups)
)

X_train = X.iloc[train_idx]
y_stable_train = y_stable[train_idx]
y_train = target_tensors[train_idx]
groups_train = groups[train_idx]

X_test = X.iloc[test_idx]
y_stable_test = y_stable[test_idx]
y_test = target_tensors[test_idx]
groups_test = groups[test_idx]

def run_loss_stage(stage_name):
    configs = LOSS_SEARCH_STAGES[stage_name]

    for i, config in enumerate(configs, start=1):
        print("\n" + "=" * 80)
        print(
            f"{stage_name}| "
            f"{i}/{len(configs)} |"
            f"{config['name']}"
        )
        print("\n" + "=" * 80)

        print("config:", config)

        cross_validation(
            X_train,
            y_train,
            y_stable_train,
            groups_train,
            ("mri", "pet", "cog", "csf", "rf"),
            run_name=f"loss/{config["name"]}",
            n_splits=10,
            random_state=RANDOM_STATE,
            imputer="median",
            scaling="min-max",
            epochs=250,
            batch_size=32,
            time_weights=config["time_weights"],
            severity_matrix=[
                [0.0, 0.5, 2.0],
                [0.5, 0.0, 1.0],
                [2.0, 1.0, 0.0]
            ],
            severity_weight=config["severity_weight"],
            transition_weight=config["transition_weight"],
            transition_loss=config["transition_loss"],
            huber_delta=config["huber_delta"],
            from_logits=False
        )
    



