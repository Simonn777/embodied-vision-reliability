"""Isolated state branches for a development-only tracking mechanism probe.

This module imports no model, tensor runtime, data reader or evaluator. The
caller binds an unchanged query and the CURRENT frame's feature grids in
``infer_current(context) -> (prediction, next_context)``. It must run that
callback in eval/no-grad mode and validate its numeric outputs. Neither ground
truth nor future frames belong in the callback. This is NOT an online selector:
only the keep branch advances the baseline, even when a shadow is better.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


def _map_context(context, transform):
    """Copy TAPIR's list-of-dictionaries structure without preserving aliases."""
    if not isinstance(context, (list, tuple)) or not context:
        raise ValueError("context must be a nonempty list/tuple of dictionaries")
    result = []
    for layer in context:
        if not isinstance(layer, dict) or not layer:
            raise ValueError("each context layer must be a nonempty dictionary")
        copied = {}
        for key, value in layer.items():
            if not isinstance(key, str) or not key:
                raise ValueError("context keys must be nonempty strings")
            copied[key] = transform(value)
        result.append(copied)
    return result


def clone_context(context):
    """Clone every tensor separately, including aliased upstream dictionary slots."""
    def clone(value):
        result = value.clone()
        if result is value:
            raise ValueError("tensor clone returned its input")
        return result
    return _map_context(context, clone)


def zero_context(context):
    """Explicit zeros with the input tensors' shapes, dtypes and devices.

    clone().zero_() is equivalent to zeros_like for supported tensor leaves;
    it avoids importing torch here and never multiplies possibly nonfinite
    values by zero. The official constructor's shared dictionary is not reused.
    """
    result = clone_context(context)
    for layer in result:
        for value in layer.values():
            value.zero_()
    return result


def _signature(context):
    def describe(value):
        return (tuple(value.shape), str(value.dtype), str(value.device))
    return _map_context(context, describe)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ShadowPrediction:
    start: int
    offset: int
    prediction: Any


@dataclass(frozen=True)
class FrameBranches:
    frame: int
    keep: Any
    zero: Any
    clone_control: Optional[Any]
    shadows: Tuple[ShadowPrediction, ...]


class StateIntervention:
    """Stream keep/zero/control and fixed zero-once shadow rollouts.

    ``validate_result(prediction, context)`` must raise on malformed/nonfinite
    model outputs, and must not mutate them. ``snapshot_prediction`` converts
    outputs into independently owned records (default: deepcopy). All mutable
    tensor state is privately cloned. The caller remains responsible for query
    identity/current-frame purity and its own model's hidden mutable buffers.

    Counts include attempted callbacks, including one which raises. A failure
    permanently poisons this controller; retrying a scientific call is not an
    implicit recovery operation. No branch from the failed frame is committed.
    """

    def __init__(self, initial_context, *, total_frames: int,
                 shadow_starts, shadow_horizon: int, control_frames,
                 validate_result: Callable, snapshot_prediction: Callable = deepcopy):
        self.total_frames = _integer(total_frames, "total_frames", 1)
        self.horizon = _integer(shadow_horizon, "shadow_horizon", 1)
        self.starts = tuple(shadow_starts)
        self.controls = tuple(control_frames)
        for values, name in ((self.starts, "shadow start"),
                             (self.controls, "control frame")):
            for value in values:
                _integer(value, name)
            if list(values) != sorted(set(values)):
                raise ValueError(f"{name}s must be sorted and unique")
        if any(start + self.horizon > self.total_frames for start in self.starts):
            raise ValueError("shadow extends beyond total_frames")
        if any(b < a + self.horizon for a, b in zip(self.starts, self.starts[1:])):
            raise ValueError("shadow windows must not overlap")
        if any(frame >= self.total_frames for frame in self.controls):
            raise ValueError("control frame outside total_frames")
        if not callable(validate_result) or not callable(snapshot_prediction):
            raise ValueError("validation and snapshot callbacks must be callable")
        self._signature = _signature(initial_context)
        self._baseline = clone_context(initial_context)
        self._active: Dict[int, Any] = {}
        self._validate = validate_result
        self._snapshot = snapshot_prediction
        self._next = 0
        self._failed = False
        self._counts = {key: 0 for key in ("keep", "zero", "clone_control", "shadow")}

    @property
    def counts(self):
        return {"completed_frames": self._next, "failed": self._failed,
                "calls": dict(self._counts),
                "total_calls": sum(self._counts.values())}

    def _call(self, kind, context, infer_current):
        # The callback may mutate the supplied context, as upstream TAPIR does.
        owned = clone_context(context)
        self._counts[kind] += 1
        output = infer_current(owned)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("inference must return (prediction, next_context)")
        prediction, next_context = output
        if _signature(next_context) != self._signature:
            raise ValueError("next context changed shape, dtype, device or keys")
        self._validate(prediction, next_context)
        return self._snapshot(prediction), clone_context(next_context)

    def step(self, frame: int, infer_current: Callable) -> FrameBranches:
        if self._failed:
            raise RuntimeError("controller failed; no further inference is allowed")
        _integer(frame, "frame")
        if frame != self._next or frame >= self.total_frames:
            raise ValueError("frames must arrive exactly once in increasing order")
        if not callable(infer_current):
            raise ValueError("infer_current must be callable")
        try:
            keep, next_baseline = self._call("keep", self._baseline, infer_current)
            zero, _ = self._call("zero", zero_context(self._baseline), infer_current)
            control = None
            if frame in self.controls:
                control, _ = self._call("clone_control", self._baseline, infer_current)
            active = dict(self._active)
            if frame in self.starts:
                active[frame] = zero_context(self._baseline)
            next_active, predictions = {}, []
            for start, state in active.items():
                prediction, next_state = self._call("shadow", state, infer_current)
                offset = frame - start
                predictions.append(ShadowPrediction(start, offset, prediction))
                if offset + 1 < self.horizon:
                    next_active[start] = next_state
            record = FrameBranches(frame, keep, zero, control, tuple(predictions))
        except BaseException:
            self._failed = True
            raise
        self._baseline = next_baseline
        self._active = next_active
        self._next += 1
        return record

    def finish(self):
        """Require the full declared stream; early prefixes are not complete runs."""
        if self._failed or self._next != self.total_frames or self._active:
            raise RuntimeError("stream incomplete or failed")
        return self.counts
