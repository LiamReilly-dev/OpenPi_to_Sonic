"""
Contract between OpenPI (sender) and the bridge node (receiver).

Units
  - Arm positions (x, y, z): meters
  - Quaternions: [qx, qy, qz, qw], unit length
  - All angles (fingers, torso, neck, legs): radians
  - Torso [roll, pitch, yaw]: Euler angles relative to the pelvis
  - Gripper/finger zero position: define for your hand (open or closed)

Notes
  - Arm poses are expressed in `frame` (default "pelvis"), `mode` is "absolute" or "delta".
  - Leg values are REFERENCES for the whole-body controller, not direct motor commands.
  - Waypoint i is meant to be executed at (chunk receive time + i * dt).
"""
import time
from dataclasses import dataclass, asdict
from typing import List

import msgpack
import numpy as np

ACTION_DIM = 55

# Single source of truth for the 55-dim layout
LAYOUT = {
    "left_arm_pose":      slice(0, 7),    # [x, y, z, qx, qy, qz, qw]
    "left_hand_fingers":  slice(7, 19),   # 12 joint angles
    "right_arm_pose":     slice(19, 26),  # [x, y, z, qx, qy, qz, qw]
    "right_hand_fingers": slice(26, 38),  # 12 joint angles
    "torso":              slice(38, 41),  # [roll, pitch, yaw]
    "neck":               slice(41, 43),  # [pan, tilt]
    "left_leg":           slice(43, 49),  # hip pitch/roll/yaw, knee, ankle pitch/roll
    "right_leg":          slice(49, 55),  # hip pitch/roll/yaw, knee, ankle pitch/roll
}
QUAT_GROUPS = ("left_arm_pose", "right_arm_pose")  # quaternion is the last 4 values

# Fail at import time if groups overlap, leave gaps, or don't sum to 55
_spans = sorted((s.start, s.stop) for s in LAYOUT.values())
assert _spans[0][0] == 0 and _spans[-1][1] == ACTION_DIM, "layout does not cover 0..ACTION_DIM"
assert all(a[1] == b[0] for a, b in zip(_spans, _spans[1:])), "layout has gaps or overlaps"
assert sum(b - a for a, b in _spans) == ACTION_DIM


@dataclass
class TrajectoryWaypoint:
    rel_time_sec: float
    left_arm_pose: List[float]       # 7
    left_hand_fingers: List[float]   # 12
    right_arm_pose: List[float]      # 7
    right_hand_fingers: List[float]  # 12
    torso: List[float]               # 3
    neck: List[float]                # 2
    left_leg: List[float]            # 6
    right_leg: List[float]           # 6


@dataclass
class OpenPIActionChunk:
    header_timestamp: float
    dt: float
    horizon_length: int
    waypoints: List[TrajectoryWaypoint]
    version: int = 1
    frame: str = "pelvis"
    mode: str = "absolute"


def _norm_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 0.5:
        raise ValueError(f"invalid quaternion (norm {n:.3f}): {q}")
    return q / n


def pack_action_chunk(raw: np.ndarray, dt: float = 0.05,
                      frame: str = "pelvis", mode: str = "absolute") -> bytes:
    """(H, 55) array -> msgpack bytes. Raises ValueError on bad input."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != ACTION_DIM:
        raise ValueError(f"bad shape {raw.shape}, expected (H, {ACTION_DIM})")
    if raw.shape[0] < 1:
        raise ValueError("empty chunk")
    if not np.isfinite(raw).all():
        raise ValueError("NaN/inf in action chunk")

    wps = []
    for i, step in enumerate(raw):
        fields = {}
        for name, sl in LAYOUT.items():
            vals = step[sl].astype(float).copy()
            if name in QUAT_GROUPS:
                vals[3:7] = _norm_quat(vals[3:7])
            fields[name] = vals.tolist()
        wps.append(TrajectoryWaypoint(rel_time_sec=float(i * dt), **fields))

    chunk = OpenPIActionChunk(header_timestamp=time.time(), dt=dt,
                              horizon_length=len(wps), waypoints=wps,
                              frame=frame, mode=mode)
    return msgpack.packb(asdict(chunk), use_bin_type=True)


def unpack_action_chunk(b: bytes) -> OpenPIActionChunk:
    """msgpack bytes -> validated chunk. Raises ValueError on anything malformed."""
    d = msgpack.unpackb(b, raw=False)
    if not isinstance(d, dict):
        raise ValueError("payload is not a map")
    if d.get("version", 1) != 1:
        raise ValueError(f"unsupported version {d.get('version')}")

    wps = [TrajectoryWaypoint(**w) for w in d["waypoints"]]
    if d["horizon_length"] != len(wps) or len(wps) < 1:
        raise ValueError("horizon_length does not match number of waypoints")
    if not (d["dt"] > 0):
        raise ValueError(f"dt must be > 0, got {d['dt']}")

    for w in wps:
        for name, sl in LAYOUT.items():
            vals = getattr(w, name)
            expected = sl.stop - sl.start
            if len(vals) != expected:
                raise ValueError(f"{name}: expected {expected} values, got {len(vals)}")
            if not np.isfinite(vals).all():
                raise ValueError(f"{name}: NaN/inf")
        for name in QUAT_GROUPS:
            _norm_quat(np.asarray(getattr(w, name)[3:7]))  # raises if degenerate

    return OpenPIActionChunk(
        header_timestamp=d["header_timestamp"],
        dt=d["dt"],
        horizon_length=d["horizon_length"],
        waypoints=wps,
        version=d.get("version", 1),
        frame=d.get("frame", "pelvis"),
        mode=d.get("mode", "absolute"),
    )


def waypoint_to_array(wp: TrajectoryWaypoint) -> np.ndarray:
    out = np.zeros(ACTION_DIM)
    for name, sl in LAYOUT.items():
        out[sl] = getattr(wp, name)
    return out


def chunk_to_array(chunk: OpenPIActionChunk) -> np.ndarray:
    """-> (H, 55) float array, the inverse of pack_action_chunk."""
    return np.stack([waypoint_to_array(w) for w in chunk.waypoints])