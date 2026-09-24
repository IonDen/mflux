from pathlib import Path

import mlx.core as mx
import pytest

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.fibo.weights.fibo_weight_mapping import FIBOWeightMapping
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_mapping import FIBOVLMWeightMapping
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
from mflux.models.qwen21.weights.qwen21_weight_mapping import Qwen21WeightMapping
from mflux.models.seedvr2.weights.seedvr2_weight_mapping import SeedVR2WeightMapping

# Tensor names read from the safetensors headers of each official checkpoint (no weights). A mapping entry
# marked required that these names cannot satisfy would reject the official checkpoint at load time.


class _OfficialNames:
    DIR = Path(__file__).parent.parent / "resources" / "checkpoint_keys"

    @staticmethod
    def weights(fixture: str) -> dict[str, mx.array]:
        names = (_OfficialNames.DIR / f"{fixture}.txt").read_text().split()
        return {name: mx.zeros((1,)) for name in names}


@pytest.mark.fast
@pytest.mark.parametrize(
    ("fixture", "mapping"),
    [
        ("flux1_schnell_vae", FluxWeightMapping.get_vae_mapping()),
        ("flux1_controlnet_canny", FluxWeightMapping.get_controlnet_transformer_mapping()),
        ("fibo_transformer", FIBOWeightMapping.get_transformer_mapping()),
        ("fibo_vlm_decoder", FIBOVLMWeightMapping.get_vlm_decoder_mapping(36)),
        ("seedvr2_7b_transformer", SeedVR2WeightMapping.get_transformer_mapping(num_blocks=36)),
        ("qwen21_text_encoder", Qwen21WeightMapping.get_text_encoder_mapping()),
    ],
)
def test_an_official_checkpoint_carries_every_required_weight(fixture, mapping):
    missing = WeightMapper.missing_required_targets(_OfficialNames.weights(fixture), mapping)

    assert [target.to_pattern for target in missing] == []


@pytest.mark.fast
def test_the_flux1_transformer_still_requires_its_output_head():
    # The ControlNet relaxation must stay inside the ControlNet mapping; FLUX.1 itself always has a head.
    required = {target.to_pattern for target in FluxWeightMapping.get_transformer_mapping() if target.required}

    assert {"proj_out.weight", "norm_out.linear.weight"} <= required


@pytest.mark.fast
def test_the_issue_748_text_encoder_layout_misses_every_required_weight():
    # mlx-community/Qwen-Image-2.1-MLX-4bit names the text encoder language_model.model.* where the official
    # checkpoint uses model.language_model.*; mflux used to load it with random weights and no error.
    official = _OfficialNames.weights("qwen21_text_encoder")
    swapped = {name.replace("model.language_model.", "language_model.model.", 1): v for name, v in official.items()}
    mapping = Qwen21WeightMapping.get_text_encoder_mapping()

    assert len(WeightMapper.missing_required_targets(swapped, mapping)) == len(mapping)
