# Jax-Baseline

Jax-Baseline is the same Reinforcement Learning implementation as any Baseline implemented through the JAX and Flax/Haiku libraries.

It's not compared to any Baseline yet, but it's two to three times faster than the Torch and Tensorflow works that we've implemented before.
Using JAX's JIT(Just In Time) compilation, we optimized a series of courses for learning and constructed them based on functions. This allows you to see how JAX's capabilities can be used effectively in various Reinforcement Learning implementations.

This implementation is configured to flexibly solve the commonly used gym and unity ml environment for testing algorithms in various complex environments.

## Installation

```
pip install -r requirement.txt
pip install .
```

## Implement log

- :heavy_check_mark: : Optional implemented
- :white_check_mark: : Defualt implemented at papers
- :x: : Not implemeted yet or can not implemented
- :zzz: : Implemented but didn't update a while (can not guarantee working well now)

### Implemented Environments

| **Name**                | **Q-Net based**    | **Actor-Critic based** | **DPG based**      |
| ----------------------- | ------------------ | ---------------------- | ------------------ |
| Gymnasium               | :heavy_check_mark: | :heavy_check_mark:     | :heavy_check_mark: |
| MultiworkerGym with Ray | :heavy_check_mark: | :heavy_check_mark:     | :heavy_check_mark: |
| Unity-ML Environments   | :zzz:              | :zzz:                  | :zzz:              |

### Implemented Algorithms

#### Q-Net bases

| **Name**    | `Per`[^PER] | `N-step`[^NSTEP][^RAINBOW]   | `NoisyNet`[^NOISY]    | `Munchausen`[^MUNCHAUSEN]       | `Ape-X`[^APEX] |
| ----------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| DQN[^DQN] | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| C51[^C51] | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QRDQN[^QRDQN]| :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| IQN[^IQN] | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| FQF[^FQF] | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| SPR[^SPR] | :white_check_mark: | :white_check_mark: | :white_check_mark: | :heavy_check_mark: | :x:                |
| BBF[^BBF] | :white_check_mark: | :white_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |

#### Actor-Critic based

| **Name**        | `Box`              | `Discrete`         | `IMPALA`[^IMPALA]    |
| --------------- | ------------------ | ------------------ | ------------------ |
| A2C[^A3C]       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO[^PPO]       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:[^APPO] |
| Truly PPO(TPPO)[^TPPO] | :heavy_check_mark: | :heavy_check_mark: | :x:                |

#### DPG bases

| **Name** | `Per`[^PER]             | `N-step`[^NSTEP][^RAINBOW]| `Ape-X`[^APEX]     |
| -------- | ----------------------- | ------------------ | ------------------ |
| DDPG[^DDPG]| :heavy_check_mark:      | :heavy_check_mark: | :heavy_check_mark: |
| TD3[^TD3]  | :heavy_check_mark:      | :heavy_check_mark: | :heavy_check_mark: |
| SAC[^SAC]  | :heavy_check_mark:      | :heavy_check_mark: | :x:                |
| TQC[^TQC]  | :heavy_check_mark:      | :heavy_check_mark: | :x:                |
| TD7[^TD7]  | :white_check_mark:(LAP) | :x:                | :x:                |

## Test

To test atari with DQN(or C51, QRQDN, IQN, FQF)

```
python test/run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 \
		--steps 5e5 --batch 32 --train_freq 1 --target_update 1000 --node 512 \
		--hidden_n 1 --final_eps 0.01 --learning_starts 20000 --gamma 0.995 --clip_rewards
```

Only 15 minutes is sufficient to run 500K steps on DQNs learning atari breakout (540 steps/sec).
This performance measurement was on Nvidia RTX3080 and AMD Ryzen 9 5950X in a single process.

```
score : 9.600, epsilon : 0.010, loss : 0.181 |: 100%|███████| 500000/500000 [15:24<00:00, 540.88it/s]
```

[^DQN]: [DQN](https://arxiv.org/abs/1312.5602v1)
[^PER]: [PER](https://arxiv.org/abs/1511.05952)
[^NSTEP]: [N-step TD](https://link.springer.com/article/10.1007/BF00115009)
[^RAINBOW]: [RAINBOW DQN](https://arxiv.org/abs/1710.02298)
[^NOISY]: [Noisy network](https://arxiv.org/abs/1706.10295)
[^MUNCHAUSEN]: [Munchausen rl](https://arxiv.org/abs/2007.14430)
[^APEX]: [Ape-X](https://arxiv.org/abs/1803.00933)
[^C51]: [C51](https://arxiv.org/abs/1707.06887)
[^QRDQN]: [QRDQN](https://arxiv.org/abs/1710.10044)
[^IQN]: [IQN](https://arxiv.org/abs/1806.06923)
[^FQF]: [FQF](https://arxiv.org/abs/1911.02140)
[^SPR]: [SPR](https://arxiv.org/abs/2007.05929)
[^BBF]: [BBF](https://arxiv.org/abs/2305.19452)
[^A3C]: [A3C](https://arxiv.org/pdf/1602.01783)
[^PPO]: [PPO](https://arxiv.org/abs/1707.06347)
[^TPPO]: [Truly PPO](https://arxiv.org/abs/1903.07940)
[^IMPALA]: [IMPALA](https://arxiv.org/abs/1802.01561)
[^APPO]: [IMPALA + PPO, APPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#appo)
[^DDPG]: [DDPG](https://arxiv.org/abs/1509.02971)
[^TD3]: [TD3](https://arxiv.org/abs/1802.09477)
[^SAC]: [SAC](https://arxiv.org/abs/1801.01290)
[^TQC]: [TQC](https://arxiv.org/abs/2005.04269)
[^TD7]: [TD7](https://arxiv.org/abs/2306.02451)