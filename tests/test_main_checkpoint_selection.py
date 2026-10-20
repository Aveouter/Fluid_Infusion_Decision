"""Regression checks for main-line checkpoint selection (synthetic values only)."""

from types import SimpleNamespace

import pytest

from train import _calc_main_score, _is_main_score_improved


def pack(f1, mae):
    return (1.0, 0.7, 0.0, f1, 0.6, 0.6, 0.8, mae, mae ** 2, {})


@pytest.mark.parametrize(
    "metric,first,second",
    [
        ("f1", pack(0.8, 30.0), pack(0.7, 20.0)),
        ("mae", pack(0.7, 20.0), pack(0.8, 30.0)),
    ],
)
def test_secondary_improvement_cannot_replace_selected_checkpoint(metric, first, second):
    args = SimpleNamespace(main_metric=metric)
    best = _calc_main_score(args, first)
    candidate = _calc_main_score(args, second)
    assert not _is_main_score_improved(candidate, best)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_score_is_never_selected(score):
    assert not _is_main_score_improved(score, -float("inf"))


def test_first_finite_score_and_ties():
    assert _is_main_score_improved(0.6, -float("inf"))
    assert not _is_main_score_improved(0.6, 0.6)
    assert not _is_main_score_improved(0.600000001, 0.6)
    assert _is_main_score_improved(0.61, 0.6)


@pytest.mark.parametrize("epsilon", [-1.0, float("nan"), float("inf")])
def test_invalid_improvement_epsilon_is_rejected(epsilon):
    with pytest.raises(ValueError):
        _is_main_score_improved(0.7, 0.6, epsilon)
