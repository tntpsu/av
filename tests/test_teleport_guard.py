import numpy as np

from av_stack import _is_teleport_jump


def test_is_teleport_jump_returns_false_when_no_previous():
    detected, distance = _is_teleport_jump(None, np.array([0.0, 0.0]), 2.0)
    assert detected is False
    assert distance == 0.0


def test_is_teleport_jump_below_threshold_is_false():
    prev = np.array([0.0, 0.0])
    curr = np.array([1.9, 0.0])
    detected, distance = _is_teleport_jump(prev, curr, 2.0)
    assert detected is False
    assert np.isclose(distance, 1.9)


def test_is_teleport_jump_above_threshold_is_true():
    prev = np.array([0.0, 0.0])
    curr = np.array([3.0, 0.0])
    detected, distance = _is_teleport_jump(prev, curr, 2.0)
    assert detected is True
    assert np.isclose(distance, 3.0)
