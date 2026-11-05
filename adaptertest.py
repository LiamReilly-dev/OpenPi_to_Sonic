"""Hardware-free checks for adapter.py. Run: python test_adapter.py"""
import threading
import time

import numpy as np
import zmq

from adapter import Adapter, ChunkPublisher, FakeObservation, FakePolicy
from bridge import Bridge, CollectSink
from contract import ACTION_DIM, LAYOUT, QUAT_GROUPS, unpack_action_chunk


class ListPublisher:
    """Captures payloads instead of using the network."""
    def __init__(self): self.payloads = []
    def publish(self, payload): self.payloads.append(payload)


class BadPolicy:
    def __init__(self, result): self.result = result
    def infer(self, obs):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make(policy, **kw):
    pub = ListPublisher()
    return Adapter(policy, FakeObservation(), pub, **kw), pub


def test_publishes_valid_chunk():
    ad, pub = make(FakePolicy(horizon=50), dt=0.02)
    assert ad.step() is True
    chunk = unpack_action_chunk(pub.payloads[0])
    assert chunk.horizon_length == 50 and chunk.dt == 0.02


def test_max_horizon():
    ad, pub = make(FakePolicy(horizon=50), max_horizon=10)
    ad.step()
    assert unpack_action_chunk(pub.payloads[0]).horizon_length == 10


def test_wrong_action_dim_publishes_nothing():
    ad, pub = make(FakePolicy(horizon=50, dim=32))
    assert ad.step() is False and pub.payloads == [] and ad.failed == 1


def test_policy_exception_publishes_nothing():
    ad, pub = make(BadPolicy(RuntimeError("server down")))
    assert ad.step() is False and pub.payloads == []


def test_nan_and_bad_quaternion_rejected():
    a = FakePolicy().infer({})["actions"]
    bad = a.copy(); bad[3, 5] = np.nan
    ad, pub = make(BadPolicy({"actions": bad}))
    assert ad.step() is False
    bad = a.copy(); bad[:, LAYOUT["left_arm_pose"].start + 3: LAYOUT["left_arm_pose"].stop] = 0.0   # zero quat
    ad, pub = make(BadPolicy({"actions": bad}))
    assert ad.step() is False and pub.payloads == []


def test_skip_latency():
    # 0.2 s inference at dt=0.05 -> about 4 waypoints already elapsed
    ad, pub = make(FakePolicy(horizon=20, latency=0.2), dt=0.05, skip_latency=True)
    ad.step()
    h = unpack_action_chunk(pub.payloads[0]).horizon_length
    assert 14 <= h <= 16, h
    ad, pub = make(FakePolicy(horizon=20), dt=0.05, skip_latency=False)
    ad.step()
    assert unpack_action_chunk(pub.payloads[0]).horizon_length == 20


def test_adapter_to_bridge_over_zmq():
    """Real sockets: Adapter -> ZMQ -> Bridge -> CollectSink, then adapter stops -> freeze."""
    port = 5598
    sink = CollectSink()
    bridge = Bridge(sink, "127.0.0.1", port, rate_hz=50)
    bstop = threading.Event()
    bt = threading.Thread(target=bridge.run, args=(bstop,), daemon=True)
    bt.start()

    pub = ChunkPublisher(port)
    ad = Adapter(FakePolicy(horizon=20), FakeObservation(), pub, dt=0.05, rate_hz=10)
    astop = threading.Event()
    at = threading.Thread(target=ad.run, args=(astop,), daemon=True)
    at.start()

    time.sleep(1.5)
    assert ad.published >= 10, ad.published
    assert len(sink.states) > 40 and sink.freezes == 0

    astop.set(); at.join(2)
    time.sleep(1.8)                                   # horizon 0.95 s + grace
    assert sink.freezes == 1, sink.freezes

    bstop.set(); bt.join(2)
    S = np.array(sink.states)
    assert np.isfinite(S).all() and S.shape[1] == ACTION_DIM
    pub.close(); bridge.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")