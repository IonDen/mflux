from pathlib import Path

import mlx.core as mx
import pytest

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_definition import FIBOVLMWeightDefinition
from mflux.models.flux.weights.flux_weight_definition import FluxControlnetWeightDefinition, FluxWeightDefinition
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
from mflux.models.seedvr2.weights.seedvr2_weight_definition import (
    SeedVR2WeightDefinition3B,
    SeedVR2WeightDefinition7B,
)

# Tensor names read from the safetensors headers of each official checkpoint (no weights), after the component's
# own prefix filter and key transform, i.e. what the loader hands the check. Captured 2026-09-24 from:
#   flux1_schnell_vae        black-forest-labs/FLUX.1-schnell @ 741f7c3ce8b3  vae/
#   flux1_controlnet_canny   InstantX/FLUX.1-dev-Controlnet-Canny @ e7cee4b2afa3  diffusion_pytorch_model.safetensors
#   fibo_transformer         briaai/FIBO @ 7c7e00ca081f  transformer/
#   fibo_vlm_decoder         briaai/FIBO-vlm @ bd13da5e0b73
#   seedvr2_7b_transformer   numz/SeedVR2_comfyUI @ 09ced7102363  seedvr2_ema_7b_fp16.safetensors
#   qwen21_text_encoder      Qwen/Qwen-Image-2.1 @ 790c92633540  text_encoder/
#   qwen21_transformer       Qwen/Qwen-Image-2.1 @ 790c92633540  transformer/
# To regenerate one, take the keys of mx.load(<file>) (lazy, reads no weights) or of the shard index's weight_map;
# only fibo_vlm_decoder is then filtered, to the "model.language_model" and "lm_head" prefixes.
# A required mapping entry these names cannot satisfy would reject the official checkpoint at load time.


class _OfficialNames:
    DIR = Path(__file__).parent.parent / "resources" / "checkpoint_keys"

    @staticmethod
    def weights(fixture: str) -> dict[str, mx.array]:
        names = (_OfficialNames.DIR / f"{fixture}.txt").read_text().split()
        return {name: mx.zeros((1,)) for name in names}


class _Component:
    @staticmethod
    def of(definition, name: str) -> ComponentDefinition:
        return next(component for component in definition.get_components() if component.name == name)

    @staticmethod
    def missing(weights: dict[str, mx.array], component: ComponentDefinition) -> list[WeightTarget]:
        # The same mapping and block/layer counts WeightLoader._load_component passes to the check.
        return WeightMapper.missing_required_targets(
            weights, component.mapping_getter(), component.num_blocks, component.num_layers
        )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("fixture", "component"),
    [
        ("flux1_schnell_vae", _Component.of(FluxWeightDefinition, "vae")),
        ("flux1_controlnet_canny", FluxControlnetWeightDefinition.get_controlnet_component()),
        ("fibo_transformer", _Component.of(FIBOWeightDefinition, "transformer")),
        ("fibo_vlm_decoder", _Component.of(FIBOVLMWeightDefinition, "decoder")),
        ("seedvr2_7b_transformer", _Component.of(SeedVR2WeightDefinition7B, "transformer")),
        ("qwen21_text_encoder", _Component.of(Qwen21WeightDefinition, "text_encoder")),
        ("qwen21_transformer", _Component.of(Qwen21WeightDefinition, "transformer")),
    ],
)
def test_an_official_checkpoint_carries_every_required_weight(fixture, component):
    missing = _Component.missing(_OfficialNames.weights(fixture), component)

    assert [target.to_pattern for target in missing] == []


@pytest.mark.fast
def test_the_flux1_transformer_still_requires_its_output_head():
    # The ControlNet relaxation must stay inside the ControlNet mapping; FLUX.1 itself always has a head.
    required = {target.to_pattern for target in FluxWeightMapping.get_transformer_mapping() if target.required}

    assert {"proj_out.weight", "norm_out.linear.weight"} <= required


@pytest.mark.fast
def test_seedvr2_3b_still_requires_its_output_ada():
    # 3B builds and applies vid_out_norm, out_shift and out_scale; only 7B (use_output_ada=False) runs without.
    component = _Component.of(SeedVR2WeightDefinition3B, "transformer")
    required = {target.to_pattern for target in component.mapping_getter() if target.required}

    assert {"vid_out_norm.weight", "out_shift", "out_scale"} <= required


@pytest.mark.fast
def test_the_issue_748_text_encoder_layout_misses_every_required_weight():
    # mlx-community/Qwen-Image-2.1-MLX-4bit names the text encoder language_model.model.* where the official
    # checkpoint uses model.language_model.*; mflux used to load it with random weights and no error.
    component = _Component.of(Qwen21WeightDefinition, "text_encoder")
    official = _OfficialNames.weights("qwen21_text_encoder")
    swapped = {name.replace("model.language_model.", "language_model.model.", 1): v for name, v in official.items()}
    required = [target for target in component.mapping_getter() if target.required]

    assert len(_Component.missing(swapped, component)) == len(required)


@pytest.mark.fast
def test_the_issue_748_transformer_rename_is_caught_on_its_own():
    # The same checkpoint stores the shared modulation layer as modulation.0.* instead of modulation.1.*. Every other
    # transformer weight matches, so this one renamed layer is the only thing the check may report.
    component = _Component.of(Qwen21WeightDefinition, "transformer")
    renamed = {
        name.replace("modulation.1.", "modulation.0.", 1): v
        for name, v in _OfficialNames.weights("qwen21_transformer").items()
    }

    missing = _Component.missing(renamed, component)

    assert [target.from_pattern for target in missing] == [["modulation.1.weight"]]
