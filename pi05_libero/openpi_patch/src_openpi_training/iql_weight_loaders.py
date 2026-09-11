from __future__ import annotations

import dataclasses
import re

import flax.traverse_util
import numpy as np

from openpi.models import model as _model
from openpi.shared import download
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class CheckpointWithIQLWeightLoader:
    """Loads π0 checkpoint and initializes missing LoRA/Q/V parameters."""

    params_path: str

    def load(
        self,
        params: at.Params,
    ) -> at.Params:
        loaded_params = _model.restore_params(
            download.maybe_download(
                self.params_path
            ),
            restore_type=np.ndarray,
        )

        return _merge_params(
            loaded_params,
            params,
            missing_regex=(
                r".*(?:lora|chunk_critic|value_network).*"
            ),
        )


def _merge_params(
    loaded_params: at.Params,
    reference_params: at.Params,
    *,
    missing_regex: str,
) -> at.Params:
    flat_reference = (
        flax.traverse_util.flatten_dict(
            reference_params,
            sep="/",
        )
    )

    flat_loaded = (
        flax.traverse_util.flatten_dict(
            loaded_params,
            sep="/",
        )
    )

    result = {}

    # Load all matching parameters from π0 base.
    for key, value in flat_loaded.items():
        if key not in flat_reference:
            continue

        reference_value = flat_reference[key]

        if value.dtype != reference_value.dtype:
            value = value.astype(
                reference_value.dtype
            )

        result[key] = value

    pattern = re.compile(
        missing_regex
    )

    # Keep freshly initialized LoRA, Q, and V parameters.
    for key, value in flat_reference.items():
        if key in result:
            continue

        if pattern.fullmatch(key):
            result[key] = value

    missing = sorted(
        set(flat_reference)
        - set(result)
    )

    if missing:
        preview = "\n".join(
            missing[:30]
        )

        raise ValueError(
            "Checkpoint loading left unexpected "
            "missing parameters:\n"
            f"{preview}"
        )

    return (
        flax.traverse_util.unflatten_dict(
            result,
            sep="/",
        )
    )
