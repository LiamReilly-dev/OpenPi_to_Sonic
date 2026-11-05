"""
Bridge node: OpenPI action chunks (contract.py, ZMQ SUB) -> 50 Hz smooth state -> a Sink.

    OpenPI --(contract, :5555)--> [ receive -> play/interpolate -> slew limit -> Sink ]

The Sink is the only part that knows about the downstream controller. Included sinks:
  print : human-readable console dump (dry run)
  zmq   : re-publishes the smoothed 50 Hz stream as msgpack on --out-port (dry run, OUR format)
  sonic : stub. Fill in once SONIC's real input protocol is confirmed.

Run:  python bridge.py --sink print
"""
import argparse
import logging
import threading
import time

import msgpack
import numpy as np
import zmq

from contract import (ACTION_DIM, LAYOUT, QUAT_GROUPS, OpenPIActionChunk,
                      chunk_to_array, unpack_action_chunk)

log = logging.getLogger("bridge")

HOST = "127.0.0.1"          # 127.0.0.1 instead of localhost avoids IPv6 issues
PORT = 5555                 # OpenPI publisher
OUT_PORT = 5556             # zmq passthrough sink
RATE_HZ = 50.0
STATUS_EVERY = 1.0
SUPPORTED_FRAMES = {"pelvis"}
SUPPORTED_MODES = {"absolute"}


# ----------------------------------------------------------------------------
# Math: blending that treats quaternion groups correctly
# ----------------------------------------------------------------------------
def slerp(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:                      # q and -q are the same rotation: take the short way
        q1, d = -q1, -d
    if d > 0.9995:                   # nearly parallel: lerp is stable and accurate
        q = q0 + w * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d)
    s0 = np.sin((1.0 - w) * th0) / np.sin(th0)
    s1 = np.sin(w * th0) / np.sin(th0)
    return s0 * q0 + s1 * q1


def blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Interpolate two 55-dim states: lerp everywhere, slerp on the quaternions."""
    out = a + (b - a) * w
    for name in QUAT_GROUPS:
        sl = LAYOUT[name]
        q = slice(sl.start + 3, sl.stop)
        out[q] = slerp(a[q], b[q], w)
    return out


# ----------------------------------------------------------------------------
# Playback of the newest chunk
# ----------------------------------------------------------------------------
class ChunkPlayer:
    """
    Turns the newest chunk into a continuous trajectory.
      - waypoint i plays at (receive time + i * dt), linearly interpolated between waypoints
      - when a new chunk replaces the old one, output blends from the last sent state
        over `blend_sec` so there is no jump
      - after the last waypoint + `grace_sec` with no new chunk, status becomes "stale"
    All times are passed in explicitly, so this is testable without sleeping.
    """

    def __init__(self, blend_sec: float = 0.1, grace_sec: float = 0.1):
        self.blend_sec = blend_sec
        self.grace_sec = grace_sec
        self.chunk: OpenPIActionChunk | None = None
        self._arr: np.ndarray | None = None
        self._rx = 0.0
        self._blend_from: np.ndarray | None = None
        self._last_out: np.ndarray | None = None

    def push(self, chunk: OpenPIActionChunk, now: float) -> None:
        self._blend_from = self._last_out
        self.chunk = chunk
        self._arr = chunk_to_array(chunk)
        self._rx = now

    def sample(self, now: float):
        """-> (state55 or None, status) with status in {"waiting", "ok", "stale"}."""
        if self.chunk is None:
            return None, "waiting"
        h, dt = self.chunk.horizon_length, self.chunk.dt
        age = now - self._rx
        if age > (h - 1) * dt + self.grace_sec:
            return None, "stale"

        pos = min(max(age / dt, 0.0), h - 1)
        i = int(pos)
        j = min(i + 1, h - 1)
        out = blend(self._arr[i], self._arr[j], pos - i)

        if self._blend_from is not None and age < self.blend_sec:
            out = blend(self._blend_from, out, age / self.blend_sec)

        self._last_out = out
        return out, "ok"


class SlewLimiter:
    """
    Caps how far each value may move per tick. Last line of defense against a bad
    chunk. Quaternions are not limited here. DEFAULTS ARE PLACEHOLDERS: tune for H2.
    """

    def __init__(self, max_pos_step: float = 0.02, max_ang_step: float = 0.1):
        self.limit = np.full(ACTION_DIM, max_ang_step)
        for name in QUAT_GROUPS:
            sl = LAYOUT[name]
            self.limit[sl.start:sl.start + 3] = max_pos_step   # xyz in meters
            self.limit[sl.start + 3:sl.stop] = np.inf          # quaternion
        self.prev: np.ndarray | None = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = state.copy()
            return state
        out = self.prev + np.clip(state - self.prev, -self.limit, self.limit)
        self.prev = out
        return out


# ----------------------------------------------------------------------------
# Sinks: everything downstream-specific lives here
# ----------------------------------------------------------------------------
class Sink:
    def send(self, state: np.ndarray, meta: dict) -> None:
        raise NotImplementedError

    def freeze(self) -> None:
        """Called once when chunks go stale, and at shutdown. Must be safe to call repeatedly."""

    def close(self) -> None:
        pass


class PrintSink(Sink):
    def __init__(self, every: float = 0.5):
        self.every, self._last = every, 0.0

    def send(self, state, meta):
        now = time.monotonic()
        if now - self._last < self.every:
            return
        self._last = now
        lines = [f"[tick] frame={meta['frame']} mode={meta['mode']}"]
        for name, sl in LAYOUT.items():
            vals = [round(float(v), 3) for v in state[sl]]
            lines.append(f"  {name:<19} ({len(vals):>2}) {vals}")
        print("\n".join(lines), flush=True)

    def freeze(self):
        print("[sink] FREEZE", flush=True)


class CollectSink(Sink):
    """Keeps everything in memory. Used by the tests."""

    def __init__(self):
        self.states, self.metas, self.freezes = [], [], 0

    def send(self, state, meta):
        self.states.append(state.copy())
        self.metas.append(meta)

    def freeze(self):
        self.freezes += 1


class ZmqPassthroughSink(Sink):
    """
    Dry-run sink: publishes the smoothed 50 Hz stream as msgpack on --out-port.
    This is OUR format, not SONIC's. Useful to check the bridge end to end with
    any local subscriber.
    """

    def __init__(self, port: int = OUT_PORT):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        log.info("passthrough sink publishing on tcp://*:%d", port)

    def send(self, state, meta):
        msg = {"t": time.time(), "frame": meta["frame"], "mode": meta["mode"]}
        for name, sl in LAYOUT.items():
            msg[name] = [float(v) for v in state[sl]]
        self.sock.send(msgpack.packb(msg, use_bin_type=True))

    def freeze(self):
        self.sock.send(msgpack.packb({"t": time.time(), "freeze": True}, use_bin_type=True))

    def close(self):
        self.sock.close(0)
        self.ctx.term()


class SonicSink(Sink):
    """
    STUB. Not wired to anything yet.

    To finish this once SONIC's input spec is confirmed:
      1. __init__: open the socket SONIC's deploy stack listens on
         (the repo's `--input-type zmq_manager`), with the right PUB/SUB roles.
      2. send(): convert the 55-dim `state` into SONIC's input representation
         (retargeting, frames, units) and frame it with SONIC's message header.
      3. freeze(): send whatever SONIC treats as hold/idle/stop.
    Until then, use --sink print or --sink zmq.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SonicSink is a stub: fill in the SONIC protocol first (see class docstring)."
        )


def make_sink(name: str, out_port: int) -> Sink:
    if name == "print":
        return PrintSink()
    if name == "zmq":
        return ZmqPassthroughSink(out_port)
    if name == "sonic":
        return SonicSink()
    raise ValueError(name)


# ----------------------------------------------------------------------------
# The node
# ----------------------------------------------------------------------------
class Bridge:
    def __init__(self, sink: Sink, host: str = HOST, port: int = PORT,
                 rate_hz: float = RATE_HZ, player: ChunkPlayer | None = None,
                 limiter: SlewLimiter | None = None):
        self.sink = sink
        self.dt = 1.0 / rate_hz
        self.player = player or ChunkPlayer()
        self.limiter = limiter or SlewLimiter()

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        # Options must be set BEFORE connect
        self.sock.setsockopt(zmq.CONFLATE, 1)         # only keep the newest chunk
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(f"tcp://{host}:{port}")
        log.info("subscribed to tcp://%s:%d (CONFLATE)", host, port)

        self.total_chunks = 0
        self.rejected = 0
        self.sent = 0

    def _poll_chunk(self):
        try:
            raw = self.sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        try:
            chunk = unpack_action_chunk(raw)
        except Exception as e:                      # network edge: never crash on bad input
            self.rejected += 1
            log.warning("rejected malformed chunk: %s", e)
            return None
        if chunk.frame not in SUPPORTED_FRAMES or chunk.mode not in SUPPORTED_MODES:
            self.rejected += 1
            log.warning("rejected chunk: unsupported frame/mode %s/%s", chunk.frame, chunk.mode)
            return None
        return chunk

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        log.info("bridge running at %.0f Hz. Ctrl+C to stop.", 1.0 / self.dt)

        next_tick = time.monotonic()
        last_status = win_start = time.monotonic()
        win_chunks, rate_hz = 0, 0.0
        frozen = False

        while not stop.is_set():
            now = time.monotonic()

            chunk = self._poll_chunk()
            if chunk is not None:
                if self.player.chunk is None:
                    log.info("first chunk: horizon=%d dt=%.3f", chunk.horizon_length, chunk.dt)
                self.player.push(chunk, now)
                self.total_chunks += 1
                win_chunks += 1

            state, status = self.player.sample(now)
            if status == "ok":
                if frozen:
                    log.info("chunks resumed")
                    frozen = False
                meta = {"frame": self.player.chunk.frame, "mode": self.player.chunk.mode}
                self.sink.send(self.limiter(state), meta)
                self.sent += 1
            elif status == "stale" and not frozen:
                log.warning("chunks stale: freezing")
                self.sink.freeze()
                frozen = True

            if now - win_start >= STATUS_EVERY:
                rate_hz, win_chunks, win_start = win_chunks / (now - win_start), 0, now
            if now - last_status >= STATUS_EVERY:
                age = "n/a" if self.player.chunk is None else f"{now - self.player._rx:.2f}s"
                log.info("%s | chunks=%d rx=%.1fHz rejected=%d sent=%d last_chunk_age=%s",
                         status.upper(), self.total_chunks, rate_hz, self.rejected, self.sent, age)
                last_status = now

            next_tick += self.dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:                                   # fell behind: resync instead of bursting
                next_tick = time.monotonic()

    def close(self) -> None:
        self.sock.close(0)
        self.ctx.term()
        self.sink.close()


def main() -> None:
    p = argparse.ArgumentParser(description="OpenPI -> SONIC bridge node")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--rate", type=float, default=RATE_HZ)
    p.add_argument("--sink", choices=["print", "zmq", "sonic"], default="print")
    p.add_argument("--out-port", type=int, default=OUT_PORT)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    sink = make_sink(args.sink, args.out_port)
    bridge = Bridge(sink, args.host, args.port, args.rate)
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass
    finally:
        sink.freeze()
        bridge.close()
        log.info("stopped. chunks=%d sent=%d rejected=%d",
                 bridge.total_chunks, bridge.sent, bridge.rejected)


if __name__ == "__main__":
    main()