# OpenPI → SONIC bridge

A small ZMQ bridge that takes action chunks from an OpenPI policy and turns them into a smooth 50 Hz stream for a whole-body controller (SONIC) on an H2 humanoid.

**Status:** the adapter and bridge run end to end against a fake policy and pass their tests. Nothing has been run against a real OpenPI server, SONIC, or a robot. `SonicSink` and `RobotObservation` are stubs.

## How it fits together

```
 GPU machine                        robot-side machine
┌────────────┐  ┌────────────┐  contract.py    ┌────────────────────────────────┐
│OpenPI      │─▶│ adapter.py │ ── ZMQ :5555 ─▶ │ bridge.py                      │ ──▶ SONIC ──▶ H2
│server (ws) │  │ (5 Hz)     │  (PUB → SUB)    │ receive → interpolate → limit  │   (via SonicSink)
└────────────┘  └────────────┘                 │ → Sink          (50 Hz)        │
                                                └────────────────────────────────┘
```

OpenPI is slow and sends a short batch of future actions a few times per second. The bridge turns those chunks into one smooth state every 20 ms and hands it to a sink.

## Files

| File | What it is |
|---|---|
| `bridge.py` | **The bridge node.** The only program you start on the robot side. |
| `adapter.py` | **The sender.** Asks an OpenPI server for action chunks and publishes them through `contract.py`. |
| `contract.py` | Shared message format (a library, not a program). Both the adapter and the bridge import it. |
| `publisher.py` | Older fake sender (dummy chunks at 10 Hz). `adapter.py --policy fake` does the same job through the real code path. |
| `test_bridge.py`, `test_adapter.py` | Hardware-free tests. |

## The contract

Each chunk is `H` waypoints spaced `dt` seconds apart (currently 10 × 0.05 s). Each waypoint is 55 numbers:

| Slice | Group | Notes |
|---|---|---|
| 0–6 | `left_arm_pose` | `[x, y, z, qx, qy, qz, qw]` |
| 7–18 | `left_hand_fingers` | 12 joint angles |
| 19–25 | `right_arm_pose` | `[x, y, z, qx, qy, qz, qw]` |
| 26–37 | `right_hand_fingers` | 12 joint angles |
| 38–40 | `torso` | `[roll, pitch, yaw]` |
| 41–42 | `neck` | `[pan, tilt]` |
| 43–48 | `left_leg` | references for the controller, not motor commands |
| 49–54 | `right_leg` | references for the controller, not motor commands |

Units: meters, radians, unit quaternions. Metadata: `frame` (`"pelvis"`), `mode` (`"absolute"`), `version` (1). Waypoint `i` is played at (time received + `i × dt`).

`contract.py` is the single source of truth for this layout. The sender calls `pack_action_chunk`, and the bridge calls `unpack_action_chunk`, which rejects anything malformed.

## Quick start (no hardware)

```bash
pip install pyzmq msgpack numpy       # Python 3.10+

python test_bridge.py                 # 7 tests, takes a few seconds
python test_adapter.py                # 7 tests

# two terminals:
python adapter.py --policy fake --obs fake    # terminal 1: fake policy -> publishes chunks
python bridge.py --sink print                 # terminal 2: the bridge
```

Bridge options: `--host`, `--port` (default 5555), `--rate` (default 50), `--sink {print,zmq,sonic}`, `--out-port` (default 5556).

## What the bridge does each tick

1. Takes the newest chunk (ZMQ `CONFLATE`, so old chunks are dropped).
2. Interpolates between waypoints: lerp everywhere, slerp for quaternions.
3. Blends from the last sent state over 0.1 s when a new chunk arrives, so there are no jumps.
4. Applies a per-tick slew limiter.
5. Sends the result to the sink.

Safety behavior:
- Chunks with an unsupported `frame` or `mode`, or that are malformed, are rejected and logged.
- If no new chunk arrives within the horizon plus 0.1 s, the bridge calls `sink.freeze()` once and resumes when chunks return.

## Sinks

The sink is the only part that knows about the downstream controller.

| Sink | Purpose |
|---|---|
| `print` | Console dump for dry runs. |
| `zmq` | Re-publishes the smoothed 50 Hz stream as msgpack on port 5556. Our own format, **not** SONIC's. |
| `sonic` | **Stub.** Raises `NotImplementedError` until implemented. |

To finish `SonicSink` (see its docstring):
1. Open the socket SONIC's deploy stack listens on.
2. Convert the 55-dim state into SONIC's input format, with retargeting, frames, units, and message header.
3. Implement `freeze()` as whatever SONIC treats as hold/idle/stop.

## Open questions and things to verify on hardware

- **SONIC's input format.** From the public docs read so far, the released SONIC checkpoints target the Unitree G1, and the default model takes an SMPL-style reference motion over ZMQ (10 future frames at 20 ms spacing), not arm poses and finger angles. H2 support was not found. Read the repo's ZMQ protocol docs and VLA Inference guide before writing `SonicSink`. The repo also hosts a "decoupled" controller (RL lower body, IK upper body) that may suit arm-pose targets better.
- **Slew limits** (0.02 m and 0.1 rad per tick) are placeholders. Tune for H2.
- **Latency.** Playback assumes zero network and inference delay.
- **Fingers** have no SMPL equivalent and need their own path.

## Setting up the real OpenPI

The openpi repo (Physical Intelligence) is run separately from this bridge, on a GPU machine. Commands below were written from memory and a quick README check: compare them to the openpi README before running.

**GPU memory (per the openpi README):** inference > 8 GB, LoRA fine-tuning > 22.5 GB, full fine-tuning > 70 GB. A 12 GB card is enough for inference. Use the larger GPUs for fine-tuning.

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git rev-parse HEAD                  # note the commit, use the same one on every machine
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
export OPENPI_DATA_HOME=/path/with/lots/of/space   # checkpoint cache
```

Smoke test with a pre-trained checkpoint on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

Notes:
- JAX grabs 75% of GPU memory by default. Lower `XLA_PYTHON_CLIENT_MEM_FRACTION` on shared machines.
- openpi is tested on Ubuntu 22.04. If another OS fights you, use its Docker setup.
- Checkpoints download from `gs://openpi-assets`. If a network blocks that, download elsewhere and copy the cache directory over, with `OPENPI_DATA_HOME` pointing at it.
- The README has a script to test inference without a robot. Use it to confirm real action chunks come out.
- Multi-GPU: set `fsdp_devices` in the training config to shard a model across GPUs.

**Typical split:** fine-tune on the big GPUs, copy the checkpoint to a smaller machine, and serve it there. OpenPI streams actions over a websocket, so the GPU can be anywhere on the network.

### The released checkpoints will not output this contract

They are fine-tuned for other platforms (DROID, ALOHA, LIBERO). To get the 55-dim H2 output you need to:
1. Collect demonstrations on the H2.
2. Convert them to a LeRobot dataset.
3. Define input/output mappings for your robot (the repo's LIBERO example shows the pattern).
4. Fine-tune a base model (`pi0_base`, `pi0_fast_base`, or `pi05_base`).

## The adapter

`adapter.py` is the program that sits between an OpenPI server and the bridge. Each cycle it gets an observation, calls `policy.infer(obs)`, takes `["actions"]`, checks it, packs it with `contract.py`, and publishes it on ZMQ port 5555.

```bash
# fake policy, no OpenPI needed
python adapter.py --policy fake --obs fake

# real server (after `serve_policy.py` is running and the observation stub is filled in)
pip install -e <openpi repo>/packages/openpi-client
python adapter.py --policy ws --policy-host <server> --policy-port 8000 --obs robot --dt 0.05
```

Options: `--dt` (seconds between waypoints; **must match the policy's control rate**), `--rate` (max chunks per second, default 5), `--max-horizon N` (publish only the first N waypoints), `--skip-latency` (drop waypoints that elapsed during inference).

Safety behavior: if the policy errors, returns the wrong shape (the action dimension must be exactly 55), or returns NaN or degenerate quaternions, the adapter publishes **nothing** for that cycle. The bridge sees the silence and freezes.

Stubs you must fill in:
- `RobotObservation`: build the observation dict your fine-tuned policy expects (images, state, prompt). The keys come from your training config.
- The websocket client call in `make_websocket_policy` was written from the openpi README and examples and has not been run against a real server. Check it against the version you install.

## Roadmap

1. Run the repo's pre-trained checkpoint end to end and look at real action chunks (check what shape and dimension come back).
2. Try `adapter.py --policy ws` against it and fix whatever the real client needs.
3. Confirm SONIC's input spec (and H2 support), then implement `SonicSink`.
4. Test in simulation before touching hardware, with an operator ready to e-stop.
5. Collect H2 demonstrations and fine-tune.