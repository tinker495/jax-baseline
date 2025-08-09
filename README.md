# Jax-Baseline

Jax-Baseline is a Reinforcement Learning implementation using JAX and Flax/Haiku libraries, mirroring the functionality of Stable-Baselines.

## Features

- 2-3 times faster than previous Torch and Tensorflow implementations
- Optimized using JAX's Just-In-Time (JIT) compilation

## Installation

```
pip install -r requirement.txt
pip install .
```

## Implementation Status

- :heavy_check_mark: : Optional implemented
- :white_check_mark: : Defualt implemented at papers
- :x: : Not implemeted yet or can not implemented

### Supported Environments

| **Name**               | **Q-Net based**    | **Actor-Critic based** | **DPG based**      |
| ---------------------- | ------------------ | ---------------------- | ------------------ |
| Gymnasium              | :heavy_check_mark: | :heavy_check_mark:     | :heavy_check_mark: |
| VectorizedGym with Ray | :heavy_check_mark: | :heavy_check_mark:     | :heavy_check_mark: |

### Implemented Algorithms

#### Q-Net bases

| **Name**      | `Double`[^DOUBLE]  | `Dueling`[^DUELING] | `Per`[^PER]        | `N-step`[^NSTEP][^RAINBOW] | `NoisyNet`[^NOISY] | `Munchausen`[^MUNCHAUSEN] | `Ape-X`[^APEX]     | `HL-Gauss`[^HL_GAUSS] |
| ------------- | ------------------ | ------------------- | ------------------ | -------------------------- | ------------------ | ------------------------- | ------------------ | --------------------- |
| DQN[^DQN]     | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:         | :heavy_check_mark: | :heavy_check_mark:        | :heavy_check_mark: | :x:                   |
| C51[^C51]     | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:         | :heavy_check_mark: | :heavy_check_mark:        | :heavy_check_mark: | :heavy_check_mark:    |
| QRDQN[^QRDQN] | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:         | :heavy_check_mark: | :heavy_check_mark:        | :heavy_check_mark: | :x:                   |
| IQN[^IQN]     | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:         | :heavy_check_mark: | :heavy_check_mark:        | :x:                | :x:                   |
| FQF[^FQF]     | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:         | :heavy_check_mark: | :heavy_check_mark:        | :x:                | :x:                   |
| SPR[^SPR]     | :white_check_mark: | :white_check_mark:  | :white_check_mark: | :white_check_mark:         | :white_check_mark: | :heavy_check_mark:        | :x:                | :heavy_check_mark:    |
| BBF[^BBF]     | :white_check_mark: | :white_check_mark:  | :white_check_mark: | :white_check_mark:         | :heavy_check_mark: | :heavy_check_mark:        | :x:                | :heavy_check_mark:    |

#### Actor-Critic based

| **Name**               | `Box`              | `Discrete`         | `IMPALA`[^IMPALA]         | `Simba`[^SIMBA] |
| ---------------------- | ------------------ | ------------------ | ------------------------- | --------------- |
| A2C[^A3C]              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:        | :x:             |
| PPO[^PPO]              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:[^APPO] | :x:             |
| Truly PPO(TPPO)[^TPPO] | :heavy_check_mark: | :heavy_check_mark: | :x:                       | :x:             |
| SPO[^SPO]              | :heavy_check_mark: | :heavy_check_mark: | :x:                       | :x:             |

#### DPG bases

| **Name**        | `Per`[^PER]                   | `N-step`[^NSTEP][^RAINBOW] | `Ape-X`[^APEX]     | `Simba`[^SIMBA]    |
| --------------- | ----------------------------- | -------------------------- | ------------------ | ------------------ |
| DDPG[^DDPG]     | :heavy_check_mark:            | :heavy_check_mark:         | :heavy_check_mark: | :heavy_check_mark: |
| TD3[^TD3]       | :heavy_check_mark:            | :heavy_check_mark:         | :heavy_check_mark: | :heavy_check_mark: |
| SAC[^SAC]       | :heavy_check_mark:            | :heavy_check_mark:         | :x:                | :heavy_check_mark: |
| DAC[^DAC]:x:    | :x:                           | :x:                        | :x:                | :x:                |
| TQC[^TQC]       | :heavy_check_mark:            | :heavy_check_mark:         | :x:                | :heavy_check_mark: |
| TD7[^TD7]       | :white_check_mark:(LAP[^LaP]) | :x:                        | :x:                | :heavy_check_mark: |
| CrossQ[^CrossQ] | :heavy_check_mark:            | :heavy_check_mark:         | :x:                | :heavy_check_mark: |
| BRO[^BRO]:x:    | :x:                           | :x:                        | :x:                | :x:                |

## Performance Compariton

- [DQN 5M](docs/dqn_5m.md)
- [DQN 100K](docs/dqn_100k.md)
- [DPG](docs/dpg_comparison.md)

## Test

To test Atari with DQN (or C51, QRDQN, IQN, FQF):

```
python test/run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 \
		--steps 5e5 --batch 32 --train_freq 1 --target_update 1000 --node 512 \
		--hidden_n 1 --final_eps 0.01 --learning_starts 20000 --gamma 0.995 --clip_rewards
```

500K steps can be run in just 15 minutes on Atari Breakout (540 steps/sec).
Performance measured on Nvidia RTX3080 and AMD Ryzen 9 5950X in a single process.

```
score : 9.600, epsilon : 0.010, loss : 0.181 |: 100%|███████| 500000/500000 [15:24<00:00, 540.88it/s]
```

[^DOUBLE]: [Double DQN paper](https://arxiv.org/abs/1509.06461)

[^DUELING]: [Dueling DQN paper](https://arxiv.org/abs/1511.06581)

[^PER]: [PER](https://arxiv.org/abs/1511.05952)

[^NSTEP]: [N-step TD](https://link.springer.com/article/10.1007/BF00115009)

[^RAINBOW]: [RAINBOW DQN](https://arxiv.org/abs/1710.02298)

[^NOISY]: [Noisy network](https://arxiv.org/abs/1706.10295)

[^MUNCHAUSEN]: [Munchausen rl](https://arxiv.org/abs/2007.14430)

[^APEX]: [Ape-X](https://arxiv.org/abs/1803.00933)

[^HL_GAUSS]: [HL-GAUSS](https://arxiv.org/abs/2403.03950)

[^DQN]: [DQN](https://arxiv.org/abs/1312.5602v1)

[^C51]: [C51](https://arxiv.org/abs/1707.06887)

[^QRDQN]: [QRDQN](https://arxiv.org/abs/1710.10044)

[^IQN]: [IQN](https://arxiv.org/abs/1806.06923)

[^FQF]: [FQF](https://arxiv.org/abs/1911.02140)

[^SPR]: [SPR](https://arxiv.org/abs/2007.05929)

[^BBF]: [BBF](https://arxiv.org/abs/2305.19452)

[^IMPALA]: [IMPALA](https://arxiv.org/abs/1802.01561)

[^SIMBA]: [SIMBA](https://arxiv.org/abs/2410.09754)

[^A3C]: [A3C](https://arxiv.org/pdf/1602.01783)

[^PPO]: [PPO](https://arxiv.org/abs/1707.06347)

[^APPO]: [IMPALA + PPO, APPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#appo)

[^TPPO]: [Truly PPO](https://arxiv.org/abs/1903.07940)

[^SPO]: [SPO](https://arxiv.org/abs/2401.16025)

[^DDPG]: [DDPG](https://arxiv.org/abs/1509.02971)

[^TD3]: [TD3](https://arxiv.org/abs/1802.09477)

[^SAC]: [SAC](https://arxiv.org/abs/1812.05905)

[^DAC]: [DAC](https://arxiv.org/abs/2310.19527)

[^TQC]: [TQC](https://arxiv.org/abs/2005.04269)

[^TD7]: [TD7](https://arxiv.org/abs/2306.02451)

[^LaP]: [LaP](https://arxiv.org/abs/2007.06049)

[^CrossQ]: [CrossQ](https://arxiv.org/abs/1902.05605)

[^BRO]: [BRO](https://arxiv.org/abs/2405.16158)
