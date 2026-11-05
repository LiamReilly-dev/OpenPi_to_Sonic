"""
Adapter: asks an OpenPI policy for action chunks and publishes them for the bridge.

    observation source --> policy.infer() --> (H, 55) --> contract.pack --> ZMQ PUB :5555 --> bridge.py

Run it next to the policy server (or anywhere that can reach it):

    python adapter.py --policy ws --policy-host <server> --policy-port 8000 --obs robot
    python adapter.py --policy fake --obs fake        # no OpenPI needed, for dry runs

If the policy fails or returns something unusable, the adapter publishes NOTHING for that
cycle. The bridge notices the silence and freezes by itself, so failing quietly is safe.

NOT TESTED against a real OpenPI server. The websocket client usage is written from the
openpi README/examples; verify it against the repo version you install.
"""
import argparse
import logging
import math
import threading
import time

import numpy as np
import zmq

from contract import ACTION_DIM, LAYOUT, QUAT_GROUPS, pack_action_chunk

log = logging.getLogger("adapter")

PUB_PORT = 5555      # the bridge subscribes here
DEFAULT_DT = 0.05    # seconds between waypoints. MUST match the policy's control rate.


# ----------------------------------------------------------------------------
# Policies: anything with .infer(obs) -> {"actions": (H, D) array}
# ----------------------------------------------------------------------------
class FakePolicy:
    """Stand-in for OpenPI: slowly moving (50, 55) chunks with valid quaternions."""

    def __init__(self, horizon: int = 50, dim: int = ACTION_DIM, latency: float = 0.0):
        self.horizon, self.dim, self.latency = horizon, dim, latency

    def infer(self, obs: dict) -> dict:
        time.sleep(self.latency)
        t = time.time() + np.arange(self.horizon) * DEFAULT_DT
        a = np.zeros((self.horizon, self.dim))
        if self.dim == ACTION_DIM:
            for name in QUAT_GROUPS:
                a[:, LAYOUT[name].start + 6] = 1.0                       # identity quaternion
            a[:, LAYOUT["left_arm_pose"].start + 2] = 0.1 + 0.05 * np.sin(t)
            a[:, LAYOUT["torso"]] = 0.02 * np.sin(t)[:, None]
        return {"actions": a}


def make_websocket_policy(host: str, port: int):
    """
    Client for `scripts/serve_policy.py`. Needs openpi's client package:
        pip install -e <openpi repo>/packages/openpi-client
    """
    from openpi_client import websocket_client_policy   # imported lazily: optional dependency
    return websocket_client_policy.WebsocketClientPolicy(host=host, port=port)


# ----------------------------------------------------------------------------
# Observation sources: anything with .get() -> dict
# ----------------------------------------------------------------------------
class FakeObservation:
    def get(self) -> dict:
        return {"prompt": "test"}


class RobotObservation:
    """
    STUB. Build the observation dict your fine-tuned policy was trained on:
    camera images, joint state, and a text prompt. The keys and shapes come from
    your training config's input mapping, so there is nothing to fill in until
    that exists. (For reference, openpi's DROID example uses keys like
    "observation/exterior_image_1_left", "observation/wrist_image_left", and "prompt".)
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RobotObservation is a stub: define it with your policy's input mapping.")


# ----------------------------------------------------------------------------
# Publisher
# ----------------------------------------------------------------------------
class ChunkPublisher:
    def __init__(self, port: int = PUB_PORT, settle_sec: float = 1.0):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        log.info("publishing chunks on tcp://*:%d", port)
        time.sleep(settle_sec)          # let subscribers connect (ZMQ slow joiner)

    def publish(self, payload: bytes) -> None:
        self.sock.send(payload)

    def close(self) -> None:
        self.sock.close(0)
        self.ctx.term()


# ----------------------------------------------------------------------------
# The adapter
# ----------------------------------------------------------------------------
class Adapter:
    def __init__(self, policy, obs_source, publisher, dt: float = DEFAULT_DT,
                 max_horizon: int | None = None, skip_latency: bool = False,
                 rate_hz: float = 5.0):
        self.policy, self.obs_source, self.publisher = policy, obs_source, publisher
        self.dt = dt
        self.max_horizon = max_horizon
        self.skip_latency = skip_latency
        self.period = 1.0 / rate_hz
        self.published = 0
        self.failed = 0

    def _prepare(self, actions, latency: float) -> np.ndarray:
        a = np.asarray(actions, dtype=float)
        if a.ndim != 2 or a.shape[1] != ACTION_DIM:
            raise ValueError(
                f"policy returned actions of shape {a.shape}; expected (H, {ACTION_DIM}). "
                f"The model's action dimension must match the contract."
            )
        if self.skip_latency:
            # The first waypoints describe moments that have already passed by the time we publish.
            skip = min(round(latency / self.dt), a.shape[0] - 1)
            a = a[skip:]
        if self.max_horizon is not None:
            a = a[: self.max_horizon]
        return a

    def step(self) -> bool:
        """One cycle: observe, infer, pack, publish. Returns True if a chunk was published."""
        try:
            obs = self.obs_source.get()
            t0 = time.monotonic()
            result = self.policy.infer(obs)
            latency = time.monotonic() - t0
            raw = self._prepare(result["actions"], latency)
            payload = pack_action_chunk(raw, self.dt)        # validates: finite, sane quaternions
        except Exception as e:                               # fail quiet: bridge freezes on silence
            self.failed += 1
            log.warning("cycle failed, nothing published: %s", e)
            return False

        self.publisher.publish(payload)
        self.published += 1
        log.info("chunk #%d: %d waypoints, inference %.0f ms", self.published, raw.shape[0], latency * 1e3)
        return True

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        log.info("adapter running (target %.1f chunks/s). Ctrl+C to stop.", 1.0 / self.period)
        while not stop.is_set():
            t0 = time.monotonic()
            self.step()
            delay = self.period - (time.monotonic() - t0)   # inference may already take longer
            if delay > 0:
                stop.wait(delay)


def main() -> None:
    p = argparse.ArgumentParser(description="OpenPI policy -> bridge adapter")
    p.add_argument("--policy", choices=["ws", "fake"], default="fake")
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, default=8000)
    p.add_argument("--obs", choices=["fake", "robot"], default="fake")
    p.add_argument("--pub-port", type=int, default=PUB_PORT)
    p.add_argument("--dt", type=float, default=DEFAULT_DT, help="seconds between waypoints (policy control rate)")
    p.add_argument("--rate", type=float, default=5.0, help="max chunks per second")
    p.add_argument("--max-horizon", type=int, default=None, help="publish only the first N waypoints")
    p.add_argument("--skip-latency", action="store_true", help="drop waypoints that elapsed during inference")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    policy = FakePolicy() if args.policy == "fake" else make_websocket_policy(args.policy_host, args.policy_port)
    obs = FakeObservation() if args.obs == "fake" else RobotObservation()
    pub = ChunkPublisher(args.pub_port)
    adapter = Adapter(policy, obs, pub, args.dt, args.max_horizon, args.skip_latency, args.rate)
    try:
        adapter.run()
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
        log.info("stopped. published=%d failed=%d", adapter.published, adapter.failed)


if __name__ == "__main__":
    main()