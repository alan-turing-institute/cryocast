"""Random Forest feature importance for input variables.

Fits one ``RandomForestRegressor`` to predict the training target from spatially-
and temporally-averaged input variables, then reads off the forest's built-in
``feature_importances_`` (mean decrease in impurity) - the single metric this
module reports.
"""

import logging

import numpy as np
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor

from icenet_mp.data import CommonDataModule

logger = logging.getLogger(__name__)


def compute_feature_importance(
    config: DictConfig,
    *,
    n_estimators: int = 500,
    random_state: int = 42,
) -> list[tuple[str, float]]:
    """Fit a Random Forest and return its feature importances, most important first.

    Uses the same training split, prediction target, and history/forecast
    configuration as ``imp train``. Each training sample is reduced to one scalar
    per input variable (its spatial and temporal mean over the sample's history
    window) and one scalar target (the spatial, temporal, and channel mean of the
    prediction target over its forecast window).

    Every configured group contributes features, including the target group's own
    variables; those lagged target features are typically the strongest predictors
    and dominate the top of the ranking.

    Args:
        config: Hydra-composed config, as passed to ``imp train``.
        n_estimators: Number of trees in the forest.
        random_state: Random seed for reproducibility.

    Returns:
        ``(feature_name, importance)`` pairs sorted by decreasing importance.

    """
    data_module = CommonDataModule(config)
    feature_names = [
        f"{group}/{variable}"
        for group, dataset in data_module.datasets.items()
        for variable in dataset.variable_names
    ]

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    # Unshuffled: the forest's bootstrap sampling is seeded by row position, so a
    # shuffled row order would make the reported importances non-reproducible.
    for batch in data_module.train_dataloader(shuffle=False):
        target = batch.pop("target")
        batch.pop("climatology", None)  # optional baseline field, not a feature
        rows.append(
            np.concatenate(
                [
                    batch[group]
                    .numpy()
                    .mean(axis=(1, 3, 4))  # (B, T, C, H, W) -> (B, C)
                    for group in data_module.datasets
                ],
                axis=1,
            )
        )
        targets.append(target.numpy().mean(axis=(1, 2, 3, 4)))  # -> (B,)

    x = np.concatenate(rows, axis=0)
    y = np.concatenate(targets, axis=0)

    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        msg = (
            "Training data contains NaN or inf values; check the configured "
            "datasets for missing cells."
        )
        raise ValueError(msg)

    logger.info("Fitting Random Forest on %d samples, %d features.", *x.shape)
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(x, y)

    ranked = sorted(
        zip(feature_names, model.feature_importances_, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [(name, float(score)) for name, score in ranked]
