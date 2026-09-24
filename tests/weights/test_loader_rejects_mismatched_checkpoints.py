import mlx.core as mx
import pytest

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.utils.exceptions import ModelConfigError


class _TinyModel:
    MAPPING = [
        WeightTarget(to_pattern="proj.weight", from_pattern=["model.proj.weight"]),
        WeightTarget(to_pattern="extra.weight", from_pattern=["model.extra.weight"], required=False),
    ]

    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [ComponentDefinition(name="encoder", hf_subdir="encoder", mapping_getter=lambda: _TinyModel.MAPPING)]

    @staticmethod
    def get_download_patterns() -> list[str]:
        return ["encoder/*.safetensors"]

    @staticmethod
    def write(root, tensors: dict[str, mx.array]) -> None:
        (root / "encoder").mkdir()
        mx.save_safetensors(str(root / "encoder" / "model.safetensors"), tensors)


@pytest.mark.fast
def test_a_checkpoint_whose_names_match_nothing_is_rejected(tmp_path):
    _TinyModel.write(tmp_path, {"other.proj.weight": mx.ones((2, 2))})

    with pytest.raises(ModelConfigError) as error:
        WeightLoader.load(weight_definition=_TinyModel, model_path=str(tmp_path))

    message = str(error.value)
    assert "encoder" in message
    assert "model.proj.weight" in message
    assert "other.proj.weight" in message


@pytest.mark.fast
def test_a_checkpoint_with_every_required_weight_loads(tmp_path):
    _TinyModel.write(tmp_path, {"model.proj.weight": mx.ones((2, 2)), "unrelated.bias": mx.ones((2,))})

    weights = WeightLoader.load(weight_definition=_TinyModel, model_path=str(tmp_path))

    assert mx.array_equal(weights.components["encoder"]["proj"]["weight"], mx.ones((2, 2))).item()
