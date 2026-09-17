# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""PolicyNormProcessor — reuse the training-time ComposedModalityTransform.

This class replaces the hand-rolled un-normalization math that previously
lived in every eval client. It rebuilds the *exact* ``ComposedModalityTransform``
used at training time from a checkpoint:

  1. Read ``config.yaml`` next to the ckpt → resolve ``data_mix`` →
     look up ``robot_type`` from ``DATASET_NAMED_MIXTURES`` →
     fetch the ``DataConfig`` from ``ROBOT_TYPE_CONFIG_MAP``.
  2. Build the transform pipeline via ``data_config.transform()``.
  3. Reconstruct a ``DatasetMetadata`` from the saved
     ``dataset_statistics.json`` (which stores **combined** per-modality
     arrays of length ``D``) by splitting it into per-key entries that match
     ``data_config.action_keys`` / ``state_keys``.
  4. ``set_metadata(...)`` binds the metadata into every transform.

After construction the caller simply invokes :meth:`unapply_actions` /
:meth:`apply_state` to get/normalize tensors using the same code path as
training — there is no second source of truth for normalization math.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    StateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.model.framework.share_tools import read_mode_config

logger = logging.getLogger(__name__)


def _resolve_robot_type(
    model_cfg: dict,
    unnorm_key: Optional[str] = None,
) -> str:
    """Look up the training robot_type from the saved cfg.

    Convention used by starVLA train scripts:
      ``cfg.datasets.vla_data.data_mix`` is a key in
      ``DATASET_NAMED_MIXTURES`` whose value is a list of
      ``(dataset_name, weight, robot_type)`` tuples.

    When a data_mix contains entries from multiple robot types (e.g.
    ``bridge_rt_1`` covers ``oxe_bridge`` + ``oxe_rt1``), ``unnorm_key``
    is used to identify which embodiment is requested.  In those mixtures
    the ``robot_type`` field of each entry **matches** the top-level key in
    ``dataset_statistics.json``, so ``unnorm_key`` serves as the selector.
    """
    try:
        data_mix = model_cfg["datasets"]["vla_data"]["data_mix"]
    except (KeyError, TypeError) as e:
        raise KeyError(
            "ckpt config.yaml is missing `datasets.vla_data.data_mix`; "
            "cannot resolve training-time robot_type."
        ) from e

    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(
            f"data_mix={data_mix!r} not in DATASET_NAMED_MIXTURES "
            f"(available: {sorted(DATASET_NAMED_MIXTURES.keys())[:20]} ...). "
            "Did you forget to register the example under examples/<bench>/train_files/data_registry/?"
        )

    mixture = DATASET_NAMED_MIXTURES[data_mix]
    robot_types = sorted({entry[2] for entry in mixture})

    if len(robot_types) == 1:
        return robot_types[0]

    # Multiple robot types in the mixture.
    # Use unnorm_key as a direct selector: for multi-robot mixtures the
    # dataset_statistics.json top-level keys equal the robot_type values.
    if unnorm_key is not None and unnorm_key in robot_types:
        return unnorm_key

    raise ValueError(
        f"data_mix={data_mix!r} contains multiple robot_types {robot_types}. "
        "Pass `unnorm_key` matching one of them to disambiguate "
        f"(e.g. unnorm_key={robot_types[0]!r})."
    )


def _infer_key_dims(
    data_config: Any,
    combined_stats: Dict[str, Any],
    modality_keys: Sequence[str],
    modality: str,
) -> Dict[str, int]:
    """Compute per-key dimensions for splitting combined stats arrays.

    Lookup priority:
      1. ``data_config.<modality>_key_dims`` — explicit dict classvar on the DataConfig
         (required for DataConfigs with non-uniform or multi-dim keys).
      2. Infer uniformly from stats array length: if ``D_total / n_keys`` is an
         integer, use that as the uniform per-key dim.
      3. Fall back to dim=1 when no stats are available (empty combined dict).

    Args:
        data_config: The DataConfig instance for the current robot type.
        combined_stats: The raw stats dict for the chosen unnorm_key (the value
            of ``norm_stats[unnorm_key]`` from ``dataset_statistics.json``).
        modality_keys: Ordered list of full keys for this modality
            (e.g. ``["action.left_joints", ...]``).
        modality: ``"action"`` or ``"state"``.

    Returns:
        Dict mapping each full key to its integer dimension.
    """
    attr = f"{modality}_key_dims"
    if hasattr(data_config, attr):
        return dict(getattr(data_config, attr))

    combined = combined_stats.get(modality, {})
    stat_arr = next((v for k, v in combined.items() if k != "mask"), None)
    n_keys = len(modality_keys)
    if stat_arr is not None and n_keys > 0:
        D_total = len(stat_arr)
        if D_total == n_keys:
            return {k: 1 for k in modality_keys}
        elif n_keys > 0 and D_total % n_keys == 0:
            dim = D_total // n_keys
            return {k: dim for k in modality_keys}
        else:
            raise ValueError(
                f"Cannot infer per-key dims for modality={modality!r}: "
                f"D_total={D_total} is not evenly divisible by n_keys={n_keys}. "
                f"Add `{attr} = {{key: dim, ...}}` to the DataConfig "
                f"(keys={list(modality_keys)})."
            )
    return {k: 1 for k in modality_keys}


def _validate_strict_stats(
    stats_for_key: Dict[str, Any],
    modality: str,
    expected_dim: int,
) -> None:
    """Validate new XPolicy checkpoint statistics without tightening legacy."""
    combined = stats_for_key.get(modality)
    if not isinstance(combined, dict):
        raise ValueError(f"Missing {modality!r} statistics for strict XPolicy schema.")

    required_fields = ("mean", "std", "min", "max", "q01", "q99")
    arrays: Dict[str, np.ndarray] = {}
    for stat_name in required_fields:
        if stat_name not in combined:
            raise ValueError(f"Missing {modality}.{stat_name} statistics.")
        values = np.asarray(combined[stat_name])
        if values.ndim != 1 or values.size != expected_dim:
            raise ValueError(
                f"{modality}.{stat_name} must be a 1D vector of length "
                f"{expected_dim}, got shape {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{modality}.{stat_name} contains NaN or Inf.")
        arrays[stat_name] = values

    if np.any(arrays["q01"] > arrays["q99"]):
        raise ValueError(f"{modality}.q01 must be <= {modality}.q99 elementwise.")

    if "mask" in combined:
        mask = np.asarray(combined["mask"])
        if mask.ndim != 1 or mask.size != expected_dim:
            raise ValueError(
                f"{modality}.mask must be a 1D vector of length {expected_dim}, "
                f"got shape {mask.shape}."
            )
        if mask.dtype != np.bool_:
            raise ValueError(f"{modality}.mask must contain booleans, got {mask.dtype}.")


def _build_dataset_metadata(
    stats_for_key: Dict[str, Any],
    embodiment_tag: Any,
    action_keys: Sequence[str],
    state_keys: Sequence[str],
    action_key_dims: Optional[Dict[str, int]] = None,
    state_key_dims: Optional[Dict[str, int]] = None,
) -> DatasetMetadata:
    """Convert the *combined* stats arrays from ``dataset_statistics.json``
    back into a per-subkey :class:`DatasetMetadata` matching what the training
    pipeline produced.

    The saved ``dataset_statistics.json`` stores stats as flat arrays of
    length ``D = sum(per_key_dims)``.  For each ``"action.<sub>"`` key we
    slice out ``dim_k`` elements starting at the current cursor and store them
    under ``statistics.action.<sub> = {"min": [v0..v_{k-1}], ...}``.

    Args:
        stats_for_key: ``norm_stats[unnorm_key]`` from ``dataset_statistics.json``.
        embodiment_tag: Embodiment tag from the DataConfig.
        action_keys: Ordered list of full action keys.
        state_keys: Ordered list of full state keys.
        action_key_dims: Per-key dimension dict (``{full_key: dim_k}``).
            Defaults to dim=1 for every key.
        state_key_dims: Per-key dimension dict for state keys.
            Defaults to dim=1 for every key.
    """
    if action_key_dims is None:
        action_key_dims = {k: 1 for k in action_keys}
    if state_key_dims is None:
        state_key_dims = {k: 1 for k in state_keys}

    def _split_combined(
        combined: Dict[str, Sequence[float]],
        keys: Sequence[str],
        key_dims: Dict[str, int],
    ):
        """Split combined arrays into per-subkey dicts using per-key dims.

        ``combined`` looks like ``{"min": [..D..], "max": [..D..], "mask": [..D..], ...}``.
        ``keys`` is the ordered list of full keys.
        ``key_dims`` maps each full key to its integer dimension.
        Returns ``(stats_per_subkey, meta_per_subkey)``.
        """
        stats_per_subkey: Dict[str, Dict[str, List[float]]] = {}
        meta_per_subkey: Dict[str, StateActionMetadata] = {}
        cursor = 0
        for full_key in keys:
            subkey = full_key.split(".", 1)[1]
            dim_k = key_dims.get(full_key, 1)
            per_key: Dict[str, List[float]] = {}
            for stat_name, arr in combined.items():
                if stat_name == "mask":
                    continue
                end = cursor + dim_k
                if end > len(arr):
                    # Saved combined array shorter than expected (truncated
                    # pad channels etc.). Skip this stat field silently.
                    continue
                per_key[stat_name] = [float(v) for v in arr[cursor:end]]
            stats_per_subkey[subkey] = per_key
            meta_per_subkey[subkey] = StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(dim_k,),
                continuous=True,
            )
            cursor += dim_k
        return stats_per_subkey, meta_per_subkey

    action_combined = stats_for_key.get("action", {})
    state_combined = stats_for_key.get("state", {})

    action_stats, action_meta = _split_combined(action_combined, action_keys, action_key_dims)

    statistics: Dict[str, Any] = {"action": action_stats}
    modalities: Dict[str, Any] = {"video": {}, "action": action_meta}
    if state_combined:
        state_stats, state_meta = _split_combined(state_combined, state_keys, state_key_dims)
        statistics["state"] = state_stats
        modalities["state"] = state_meta

    return DatasetMetadata.model_validate(
        {
            "statistics": statistics,
            "modalities": modalities,
            "embodiment_tag": embodiment_tag.value if hasattr(embodiment_tag, "value") else embodiment_tag,
        }
    )


class PolicyNormProcessor:
    """Server-side normalization helper backed by training-time transforms.

    Construct once per checkpoint; call :meth:`unapply_actions` to convert
    a normalized action chunk ``(T, D)`` back to env-space.

    Args:
        ckpt_path: Path to the ``*.pt`` checkpoint (the loader looks for
            ``config.yaml`` and ``dataset_statistics.json`` two dirs up).
        unnorm_key: Which top-level key in ``dataset_statistics.json`` to use.
            ``None`` → auto-pick the only key.
    """

    def __init__(self, ckpt_path: str, unnorm_key: Optional[str] = None) -> None:
        self._ckpt_path = str(ckpt_path)
        cfg, norm_stats = read_mode_config(self._ckpt_path)
        self._model_cfg = cfg
        self._norm_stats = norm_stats

        # 3-early) Pick the requested unnorm_key (or auto-select) BEFORE
        # resolving robot_type so we can use it as a hint for multi-robot mixtures.
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            # else: defer error to step 3 below after robot_type resolution attempt
        elif unnorm_key not in norm_stats:
            raise KeyError(
                f"unnorm_key={unnorm_key!r} not in {list(norm_stats.keys())}"
            )
        self._unnorm_key = unnorm_key  # may still be None for multi-key case

        # 1) Resolve which DataConfig was used at training.
        robot_type = _resolve_robot_type(cfg, unnorm_key=unnorm_key)
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise KeyError(
                f"robot_type={robot_type!r} not in ROBOT_TYPE_CONFIG_MAP "
                f"(available: {sorted(ROBOT_TYPE_CONFIG_MAP.keys())}). "
                "Make sure the example's data_registry/data_config.py is importable."
            )
        self._robot_type = robot_type
        self._data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        self._action_keys: List[str] = list(self._data_config.action_keys)
        self._state_keys: List[str] = list(getattr(self._data_config, "state_keys", []))

        # 2) Build training-time transform pipeline.
        transform = self._data_config.transform()
        if not isinstance(transform, ComposedModalityTransform):
            transform = ComposedModalityTransform(transforms=[transform])
        self._transform = transform

        # 3) Pick the requested unnorm_key (finalize; error if still None here).
        if self._unnorm_key is None:
            raise ValueError(
                f"Multiple unnorm_keys in dataset_statistics.json: "
                f"{list(norm_stats.keys())}. Pass unnorm_key explicitly."
            )
        unnorm_key = self._unnorm_key

        # 4) Resolve per-key dims (handles multi-d action/state keys).
        stats_for_unnorm = norm_stats[unnorm_key]
        self._action_key_dims: Dict[str, int] = _infer_key_dims(
            self._data_config, stats_for_unnorm, self._action_keys, "action"
        )
        self._state_key_dims: Dict[str, int] = _infer_key_dims(
            self._data_config, stats_for_unnorm, self._state_keys, "state"
        )

        # 5) Build & bind metadata.
        ds_meta = _build_dataset_metadata(
            stats_for_key=stats_for_unnorm,
            embodiment_tag=self._data_config.embodiment_tag,
            action_keys=self._action_keys,
            state_keys=self._state_keys,
            action_key_dims=self._action_key_dims,
            state_key_dims=self._state_key_dims,
        )
        self._transform.set_metadata(ds_meta)
        self._transform.eval()  # mark transforms as eval-mode

        logger.info(
            "PolicyNormProcessor ready: robot_type=%s, unnorm_key=%s, "
            "action_keys=%s (dims=%s), state_keys=%s",
            robot_type,
            unnorm_key,
            self._action_keys,
            [self._action_key_dims[k] for k in self._action_keys],
            self._state_keys,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def action_keys(self) -> List[str]:
        return list(self._action_keys)

    @property
    def state_keys(self) -> List[str]:
        return list(self._state_keys)

    @property
    def action_key_dims(self) -> Dict[str, int]:
        return dict(self._action_key_dims)

    @property
    def state_key_dims(self) -> Dict[str, int]:
        return dict(self._state_key_dims)

    @property
    def unnorm_key(self) -> str:
        return self._unnorm_key

    @property
    def available_unnorm_keys(self) -> List[str]:
        return list(self._norm_stats.keys())

    @property
    def action_dim(self) -> int:
        return sum(self._action_key_dims.values())

    @property
    def state_dim(self) -> int:
        return sum(self._state_key_dims.values())

    @property
    def robot_type(self) -> str:
        return self._robot_type

    @property
    def transform(self) -> ComposedModalityTransform:
        return self._transform

    def validate_xpolicylab_schema(
        self,
        schema: Dict[str, Any],
        *,
        model_action_dim: int,
        model_state_dim: int,
    ) -> None:
        """Close checkpoint → live registry → normalization-stat contracts.

        This strict path is invoked only for new XPolicy schema version 1
        checkpoints. Legacy StarVLA/RoboDojo checkpoints keep their previous
        compatibility behavior.
        """
        if not isinstance(schema, dict) or schema.get("version") != 1:
            raise ValueError(f"Unsupported XPolicy normalization schema: {schema!r}.")
        if schema.get("robot_type") != self._robot_type:
            raise ValueError(
                f"Checkpoint robot_type={schema.get('robot_type')!r} does not match "
                f"live registry robot_type={self._robot_type!r}."
            )

        live_state = [
            {"key": key, "dim": int(self._state_key_dims[key])}
            for key in self._state_keys
        ]
        live_action = [
            {"key": key, "dim": int(self._action_key_dims[key])}
            for key in self._action_keys
        ]
        if schema.get("state") != live_state:
            raise ValueError(
                f"Checkpoint state schema {schema.get('state')!r} does not match "
                f"live registry {live_state!r}."
            )
        if schema.get("action") != live_action:
            raise ValueError(
                f"Checkpoint action schema {schema.get('action')!r} does not match "
                f"live registry {live_action!r}."
            )

        live_mode = getattr(self._data_config, "normalization_mode", None)
        if schema.get("normalization_mode") != live_mode:
            raise ValueError(
                f"Checkpoint normalization_mode={schema.get('normalization_mode')!r} "
                f"does not match live registry mode={live_mode!r}."
            )
        if model_action_dim is None or model_state_dim is None:
            raise ValueError("XPolicy checkpoint model config must declare action_dim and state_dim.")
        if int(model_action_dim) != self.action_dim or int(model_state_dim) != self.state_dim:
            raise ValueError(
                "Model head dimensions do not match checkpoint normalization schema: "
                f"model action/state={model_action_dim}/{model_state_dim}, "
                f"schema action/state={self.action_dim}/{self.state_dim}."
            )

        source_action_keys = schema.get("source_action_keys")
        source_action_component_indices = schema.get(
            "source_action_component_indices"
        )
        live_source_action_keys = getattr(
            self._data_config, "source_action_keys", None
        )
        live_source_action_component_indices = getattr(
            self._data_config, "source_action_component_indices", None
        )
        if source_action_keys != live_source_action_keys:
            raise ValueError(
                f"Checkpoint source_action_keys={source_action_keys!r} does not "
                f"match live registry {live_source_action_keys!r}."
            )
        if source_action_component_indices != live_source_action_component_indices:
            raise ValueError(
                "Checkpoint source_action_component_indices do not match the "
                f"live registry: checkpoint={source_action_component_indices!r}, "
                f"registry={live_source_action_component_indices!r}."
            )

        stats_for_key = self._norm_stats[self._unnorm_key]
        _validate_strict_stats(stats_for_key, "action", self.action_dim)
        _validate_strict_stats(stats_for_key, "state", self.state_dim)

    # ------------------------------------------------------------------
    # Forward path (env state -> model input)
    # ------------------------------------------------------------------
    def apply_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize proprioceptive state with the training-time transform."""
        if not self._state_keys:
            return np.asarray(state, dtype=np.float32)

        state_arr = np.array(state, dtype=np.float32, copy=True)
        original_ndim = state_arr.ndim
        if state_arr.ndim == 1:
            state_arr = state_arr[None, :]
        if state_arr.ndim != 2:
            raise ValueError(f"Expected state shape (T, D) or (D,), got {state_arr.shape}")

        data: Dict[str, np.ndarray] = {}
        cursor = 0
        for full_key in self._state_keys:
            dim_k = self._state_key_dims.get(full_key, 1)
            data[full_key] = state_arr[..., cursor : cursor + dim_k]
            cursor += dim_k

        if cursor != state_arr.shape[-1]:
            raise ValueError(
                f"Sum of per-key dims ({cursor}) != state_dim ({state_arr.shape[-1]}). "
                f"state_keys={self._state_keys}, state_key_dims={self._state_key_dims}"
            )

        out = self._transform.apply(data)
        parts: List[np.ndarray] = []
        for full_key in self._state_keys:
            value = out[full_key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value, dtype=np.float32))
        normalized = np.concatenate(parts, axis=-1)
        return normalized[0] if original_ndim == 1 else normalized

    # ------------------------------------------------------------------
    # Inverse path (model output → env action)
    # ------------------------------------------------------------------
    def unapply_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Invert action normalization using the training-time pipeline.

        Args:
            normalized_actions: shape ``(T, D)`` where
                ``D == sum(action_key_dims.values())``.

        Returns:
            ``(T, D)`` un-normalized actions in env coordinates.
        """
        normalized_actions = np.asarray(normalized_actions)
        assert normalized_actions.ndim == 2, (
            f"Expected (T, D); got shape {normalized_actions.shape}"
        )

        # Split (T, D) into per-key {full_key: torch.Tensor[T, dim_k]}.
        data: Dict[str, torch.Tensor] = {}
        cursor = 0
        for full_key in self._action_keys:
            dim_k = self._action_key_dims.get(full_key, 1)
            slice_ = normalized_actions[..., cursor : cursor + dim_k]
            data[full_key] = torch.as_tensor(slice_, dtype=torch.float32)
            cursor += dim_k

        if cursor != normalized_actions.shape[-1]:
            raise ValueError(
                f"Sum of per-key dims ({cursor}) != action_dim "
                f"({normalized_actions.shape[-1]}). "
                f"action_keys={self._action_keys}, "
                f"action_key_dims={self._action_key_dims}"
            )

        out = self._transform.unapply(data)

        parts: List[np.ndarray] = []
        for full_key in self._action_keys:
            v = out[full_key]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            parts.append(np.asarray(v))
        return np.concatenate(parts, axis=-1)
