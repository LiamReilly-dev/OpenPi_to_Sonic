"""
Hardware-free checks for the bridge. Run:  python test_bridge.py   (or: pytest -q)
"""
import threading
import time

import numpy as np

from contract import (ACTION_DIM, LAYOUT, QUAT_GROUPS, chunk_to_array,
                      pack_action_chunk, unpack_action_chunk)
from bridge import Bridge, ChunkPlayer, CollectSink, SlewLimiter, slerp

DT = 0.05


def make_raw(h=10):
    """Identity quaternions; left-arm x ramps 0, .01, .02, ..."""
    raw = np.zeros((h, ACTION_DIM))
    for name in QUAT_GROUPS:
        raw[:, LAYOUT[name].start + 6] = 1.0
    raw[:, LAYOUT["left_arm_pose"].start] = np.arange(h) * 0.01
    return raw


def make_chunk(raw=None, **kw):
    return unpack_action_chunk(pack_action_chunk(make_raw() if raw is None else raw, DT, **kw))


def test_roundtrip():
    raw = make_raw()
    raw[:, LAYOUT["torso"]] = 0.123
    out = chunk_to_array(make_chunk(raw))
    assert out.shape == raw.shape and np.allclose(out, raw)


def test_rejects_bad_input():
    bad = make_raw()
    bad[3, 5] = np.nan
    for args in [bad, np.zeros((10, 54))]:
        try:
            pack_action_chunk(args)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    try:
        unpack_action_chunk(b"\x00")
    except Exception:
        pass
    else:
        raise AssertionError("garbage should not unpack")


def test_slerp():
    q = np.array([0, 0, 0, 1.0])
    r = np.array([0, 0, np.sin(0.5), np.cos(0.5)])          # 1 rad about z
    assert np.allclose(slerp(q, r, 0.0), q) and np.allclose(slerp(q, r, 1.0), r)
    mid = slerp(q, r, 0.5)
    assert abs(np.linalg.norm(mid) - 1) < 1e-9
    assert np.allclose(mid, [0, 0, np.sin(0.25), np.cos(0.25)])
    assert abs(abs(np.dot(slerp(q, -q, 0.5), q)) - 1) < 1e-9   # q and -q: no wild swing


def test_player_interpolation_and_stale():
    p = ChunkPlayer(grace_sec=0.1)
    assert p.sample(0.0) == (None, "waiting")
    p.push(make_chunk(), now=100.0)
    x = LAYOUT["left_arm_pose"].start
    s, st = p.sample(100.0 + 1.5 * DT)
    assert st == "ok" and abs(s[x] - 0.015) < 1e-9
    s, st = p.sample(100.0 + 9 * DT)                          # last waypoint
    assert st == "ok" and abs(s[x] - 0.09) < 1e-9
    assert p.sample(100.0 + 9 * DT + 0.05)[1] == "ok"         # inside grace
    assert p.sample(100.0 + 9 * DT + 0.2)[1] == "stale"


def test_player_blends_between_chunks():
    p = ChunkPlayer(blend_sec=0.1)
    p.push(make_chunk(), now=0.0)
    last, _ = p.sample(0.2)                                   # x = 0.04
    new = make_raw()
    new[:, LAYOUT["left_arm_pose"].start] = 1.0               # big jump
    p.push(make_chunk(new), now=0.2)
    x = LAYOUT["left_arm_pose"].start
    assert abs(p.sample(0.2)[0][x] - last[x]) < 1e-9          # starts where we were
    assert abs(p.sample(0.3)[0][x] - 1.0) < 1e-9              # fully on the new chunk


def test_slew_limiter():
    lim = SlewLimiter(max_pos_step=0.02, max_ang_step=0.1)
    a = make_raw(1)[0]
    lim(a)
    b = a.copy()
    b[LAYOUT["torso"].start] = 5.0
    b[0] = 1.0
    out = lim(b)
    assert abs(out[LAYOUT["torso"].start] - 0.1) < 1e-9
    assert abs(out[0] - 0.02) < 1e-9
    assert np.allclose(out[3:7], b[3:7])                      # quaternion untouched


def test_end_to_end_localhost():
    """Real ZMQ: publisher -> Bridge -> CollectSink, then publisher dies -> freeze."""
    from publisher import OpenPIPublisher, make_dummy_chunk
    port = 5599
    sink = CollectSink()
    bridge = Bridge(sink, "127.0.0.1", port, rate_hz=50)
    stop = threading.Event()
    th = threading.Thread(target=bridge.run, args=(stop,), daemon=True)
    th.start()

    pub = OpenPIPublisher(port)                               # sleeps 1 s for the join
    t_end = time.time() + 1.5
    while time.time() < t_end:
        pub.send_action_chunk(make_dummy_chunk())
        time.sleep(0.1)
    n_live = len(sink.states)
    assert n_live > 40, n_live                                # ~50 Hz for ~1.5 s
    assert sink.freezes == 0

    time.sleep(1.0)                                           # publisher silent
    assert sink.freezes == 1, sink.freezes
    assert len(sink.states) < n_live + 40                     # no output while frozen

    pub.send_action_chunk(make_dummy_chunk())                 # recovery
    time.sleep(0.3)
    assert len(sink.states) > n_live + 5

    stop.set()
    th.join(2)
    S = np.array(sink.states)
    assert np.isfinite(S).all() and S.shape[1] == ACTION_DIM
    assert (np.abs(np.diff(S, axis=0)) <= SlewLimiter().limit + 1e-9).all()
    pub.sock.close(0); pub.ctx.term(); bridge.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")