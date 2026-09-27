# FlashSAC

jax-baseline's FlashSAC (JAX/Flax) against the reference PyTorch implementation
([Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC/tree/87edc9061150ae9e962dd84e6544e27a1554b3ab)).
The comparison covers two regimes: a CPU-simulated single environment (Gym MuJoCo Humanoid-v4)
and a GPU-simulated batch of 1024 environments (mjlab Unitree G1).

On this machine, the JAX version finishes the same training budget faster in both regimes:

- Humanoid-v4, 1M steps: 51% faster.
- mjlab G1, 50M steps: 10% faster.

Final returns are within the run-to-run spread. Each implementation was run with one seed.

## Humanoid-v4 (Gym MuJoCo, CPU simulation, 1 environment)

![FlashSAC Humanoid-v4](figures/flashsac_humanoid_v4_1m.png)

| Implementation              | Wall-clock to 1M steps | Train throughput (env steps/s) | Time to 90% of paper (11,345) | Final eval @1M         |
| --------------------------- | ---------------------- | ------------------------------ | ----------------------------- | ---------------------- |
| Upstream FlashSAC (PyTorch) | 2.07 h                 | 144                            | 1.03 h                        | 12,849                 |
| jax-baseline FlashSAC (JAX) | **1.01 h (−51%)**      | **294**                        | **0.40 h**                    | 12,955                 |
| Paper (5 seeds)             | –                      | –                              | –                             | 12,606 (12,329–12,713) |

- Setup:
  - Seed 0, 1M env steps, batch 512, one update per env step, learning starts at 10k steps.
  - Evaluation: 50 episodes every 100k steps.
  - Upstream uses its `scripts/run_mujoco.sh` recipe with `torch.compile` and AMP.
  - JAX uses [`experiments/configs/mujoco/flashsac_humanoid.yaml`](../experiments/configs/mujoco/flashsac_humanoid.yaml) at commit `6f0279f6`.
- Throughput is summed over the evaluation-free intervals after 20k steps. Wall-clock is measured from launch to the logged final evaluation.
- This JAX seed learned slowly before 300k steps (1,321 at 100k and 1,954 at 200k). It also dipped at the 900k evaluation (8,844).
  - Replaying that evaluation's 50 initial states with every run's final policy gave no falls, on both the CPU and GPU action paths. So the dip belongs to the 900k policy, not to the evaluation or the action path.
  - The paper's 5-seed band shows dips of the same kind (down to 4.7k at 600k).

## mjlab Unitree G1 flat velocity (GPU simulation, 1024 environments)

![FlashSAC mjlab G1](figures/flashsac_mjlab_g1_50m.png)

| Implementation                          | Wall-clock to final eval | Train throughput (env steps/s) | Time to 62.3 (90% of best final) | Final eval | Best eval |
| --------------------------------------- | ------------------------ | ------------------------------ | -------------------------------- | ---------- | --------- |
| Upstream FlashSAC (PyTorch, mjlab port) | 0.75 h                   | 20,978                         | 0.54 h                           | 69.2\*     | 69.2      |
| jax-baseline FlashSAC (JAX)             | **0.68 h (−10%)**        | **23,201**                     | **0.49 h**                       | 63.8       | 67.9      |

\* Upstream logs the mean of its last two evaluations (the periodic one at 49.99M steps and the final one). The same statistic for the JAX run is 65.9.

- Setup:
  - Task `Mjlab-Velocity-Flat-Unitree-G1`, 1024 environments, actor observations only, seed 0, 50M env steps.
  - Two updates per vector step, batch 2048, 3-step returns, 5M-transition GPU replay.
  - Evaluation: 1024 episodes, 10 times per run.
- Upstream: the reference implementation has no mjlab support. The baseline is a local port of it that mirrors its IsaacLab wrapper, run with `torch.compile` and AMP.
  - Like that wrapper, the port uses the reset observation in place of the terminal observation at time-outs. The JAX adapter keeps the true terminal observation.
- JAX: [`experiments/configs/mjlab/flashsac_mjlab_g1.yaml`](../experiments/configs/mjlab/flashsac_mjlab_g1.yaml) at commit `d273c23e`.
- Throughput is summed over the evaluation-free intervals after 0.5M steps.

## What makes the JAX version fast

The algorithm follows FlashSAC:

- Categorical twin critics (101 atoms) run as one vmapped ensemble.
- The encoders use BatchNorm and RMSNorm, with unit-norm weight projection.
- Exploration noise is repeated for a number of steps drawn from a zeta distribution.
- The reward divisor is `max(std(G), max|G| / G_max)`.
- The entropy target is set from `sigma_target`.

The speed comes from how each step is executed:

- **Compiled hot paths:**
  - Action selection, updates, reward normalization and replay batch preparation are each one compiled call.
  - Transfers between host and device are explicit; `--strict_transfers` fails on any implicit one.
- **Bulk updates** (`train_freq` and `gradient_steps` both 32): 32 updates run as one compiled scan every 32 env steps, keeping one update per env step.
  - The acting policy can be up to 32 updates behind, where upstream updates after every step.
- **Host-simulated envs:**
  - Actions are computed on the CPU backend from a mirror of the policy parameters. The mirror refreshes only when the parameters change. Under WSL2 this took 0.25 ms per step, against 2.0 ms for the GPU round trip.
  - Reward-normalizer records are queued and applied in one upload when the statistics are next read.
- **Device-simulated envs (mjlab):**
  - Observations reach JAX through DLPack without a host copy.
  - The flashbax GPU replay fuses sample → batch preparation → updates into one call, and record → episode statistics → add into another.
  - The update pulse is dispatched after the action and before the simulator step, so it overlaps the simulator. The reward normalizer it reads lags by one vector step.
  - Metrics are logged every 100 vector steps (`log_interval: 102400`).
  - Only the observation group in use is built.
- **Persistent compilation cache** (`~/.cache/jax_baselines/xla`): repeated runs skip recompiling the update step.

## Reproduce

```bash
uv run exp experiments/configs/mujoco/flashsac_humanoid.yaml --set logger=tensorboard
uv run exp experiments/configs/mjlab/flashsac_mjlab_g1.yaml --set env_observation_key=actor --set eval_eps=1024 --set logger=tensorboard
```

- Hardware: RTX 4080 SUPER 16 GB and Ryzen 7 7800X3D, under WSL2, with runs pinned to CPUs 0–7.
- WSL2 makes GPU round trips expensive, which inflates the gain from acting on the CPU. On native Linux the Humanoid gap may be smaller.
- These are single-seed runs. Differences in final return between the implementations are within the run-to-run spread.
