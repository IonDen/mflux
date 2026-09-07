import pytest
from mlx import nn

from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.saving.model_saver import ModelSaver
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class _TinyComponent(nn.Module):
    # Two Linears whose last dim is a multiple of 64, so the definition's predicate
    # quantizes them and the save/load path carries real quantized tensors.
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 64)
        self.b = nn.Linear(64, 64)


class _SharedSubdirDefinition:
    # The SeedVR2 layout: two non-tokenizer components that both sit flat at repo root
    # (hf_subdir="."). On the real repo, weight_files tells them apart on load. On save,
    # both would write 0.safetensors and model.safetensors.index.json into the same
    # directory, so the second clobbers the first (issue #621).
    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(name="transformer", hf_subdir=".", loading_mode="mlx_native"),
            ComponentDefinition(name="vae", hf_subdir=".", loading_mode="mlx_native"),
        ]

    @staticmethod
    def get_tokenizers() -> list:
        return []

    @staticmethod
    def get_download_patterns() -> list[str]:
        return ["**/*.safetensors"]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0


class TestModelSaverSharedSubdir:
    @pytest.mark.fast
    def test_two_components_sharing_a_subdir_roundtrip_without_collision(self, tmp_path):
        # RED without the fix: transformer and vae both save into the same directory
        # (hf_subdir="."), so the second overwrites the first's shards and index, and on
        # reload the transformer comes back with the VAE's weights. GREEN once ModelSaver
        # and WeightLoader give each shared-subdir component its own <subdir>/<name>
        # directory. Issue #621.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=_SharedSubdirDefinition,
            make_components=lambda: {"transformer": _TinyComponent(), "vae": _TinyComponent()},
            base_path=tmp_path / "shared_subdir_tiny_q8",
            bits=8,
        )

    @pytest.mark.fast
    def test_two_written_components_in_the_same_directory_raise(self, tmp_path):
        # If a definition ever writes two components into one directory, the second would
        # silently overwrite the first's shards and index (#621); ModelSaver must fail loudly.
        class _FilteredPairDefinition:
            @staticmethod
            def get_components():
                return [
                    ComponentDefinition(name="a", hf_subdir="", weight_prefix_filters=["x"]),
                    ComponentDefinition(name="b", hf_subdir="", weight_prefix_filters=["y"]),
                ]

            @staticmethod
            def get_tokenizers():
                return []

        class _Model:
            def __init__(self):
                self.a = _TinyComponent()
                self.b = _TinyComponent()

        with pytest.raises(ValueError, match="both save to"):
            ModelSaver.save_model(
                model=_Model(),
                bits=8,
                base_path=str(tmp_path / "collide"),
                weight_definition=_FilteredPairDefinition,
            )

    def test_independent_components_spelled_empty_and_dot_do_not_collide(self):
        # "" and "." are the same directory. Two independent components spelled differently
        # must still be separated, not left to overwrite each other (#621).
        components = [
            ComponentDefinition(name="a", hf_subdir=""),
            ComponentDefinition(name="b", hf_subdir="."),
        ]
        subdirs = ComponentDefinition.save_subdirs(components)
        from pathlib import Path

        assert str(Path(subdirs["a"])) != str(Path(subdirs["b"]))

    def test_guard_catches_a_collision_spelled_empty_and_dot(self, tmp_path):
        # The save-time guard compares resolved directories, so a shared-source pair spelled
        # "" and "." (which save_subdirs leaves in place) is still caught, not missed (#621).
        class _FilteredDotPairDefinition:
            @staticmethod
            def get_components():
                return [
                    ComponentDefinition(name="a", hf_subdir="", weight_prefix_filters=["x"]),
                    ComponentDefinition(name="b", hf_subdir=".", weight_prefix_filters=["y"]),
                ]

            @staticmethod
            def get_tokenizers():
                return []

        class _Model:
            def __init__(self):
                self.a = _TinyComponent()
                self.b = _TinyComponent()

        with pytest.raises(ValueError, match="both save to"):
            ModelSaver.save_model(
                model=_Model(),
                bits=8,
                base_path=str(tmp_path / "collide_dot"),
                weight_definition=_FilteredDotPairDefinition,
            )
