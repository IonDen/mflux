import mlx.core as mx
import pytest

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.common.weights.mapping.weight_mapping import WeightTarget


class _Weights:
    @staticmethod
    def named(*names: str) -> dict[str, mx.array]:
        return {name: mx.zeros((1,)) for name in names}


@pytest.mark.fast
def test_a_required_target_with_no_source_tensor_is_reported():
    mapping = [
        WeightTarget(to_pattern="a.weight", from_pattern=["a.weight"]),
        WeightTarget(to_pattern="b.weight", from_pattern=["b.weight"]),
    ]

    missing = WeightMapper.missing_required_targets(_Weights.named("a.weight"), mapping)

    assert [target.to_pattern for target in missing] == ["b.weight"]


@pytest.mark.fast
def test_an_optional_target_is_never_reported():
    mapping = [WeightTarget(to_pattern="b.weight", from_pattern=["b.weight"], required=False)]

    assert WeightMapper.missing_required_targets(_Weights.named("a.weight"), mapping) == []


@pytest.mark.fast
def test_one_matching_block_satisfies_a_block_pattern():
    # The mapper expands every block pattern to the largest block count it detects, so FLUX.1's 19 double
    # blocks are looked for up to 38. Counting phantom blocks as missing would reject every FLUX.1 checkpoint.
    mapping = [
        WeightTarget(
            to_pattern="transformer_blocks.{block}.attn.to_q.weight",
            from_pattern=["transformer_blocks.{block}.attn.to_q.weight"],
        )
    ]
    weights = _Weights.named("transformer_blocks.0.attn.to_q.weight", "single_transformer_blocks.37.proj_out.weight")

    assert WeightMapper.missing_required_targets(weights, mapping) == []


@pytest.mark.fast
def test_any_alternative_source_name_satisfies_a_target():
    mapping = [WeightTarget(to_pattern="norm.weight", from_pattern=["old.norm.weight", "new.norm.weight"])]

    assert WeightMapper.missing_required_targets(_Weights.named("new.norm.weight"), mapping) == []
