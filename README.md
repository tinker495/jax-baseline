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

| **Name** | `Per`              | `N-step`           | `NoisyNet`         | `Munchausen`       | `Ape-X`            |
| -------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| C51      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QRDQN    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| IQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| FQF      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| SPR      | :white_check_mark: | :white_check_mark: | :white_check_mark: | :heavy_check_mark: | :x:                |
| BBF      | :white_check_mark: | :white_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |

#### Actor-Critic based

| **Name**        | `Box`              | `Discrete`         | `IMPALA`           |
| --------------- | ------------------ | ------------------ | ------------------ |
| A2C             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Truly PPO(TPPO) | :heavy_check_mark: | :heavy_check_mark: | :x:                |

#### DPG bases

| **Name** | `Per`                   | `N-step`           | `Ape-:x:`          |
| -------- | ----------------------- | ------------------ | ------------------ |
| DDPG     | :heavy_check_mark:      | :heavy_check_mark: | :heavy_check_mark: |
| TD3      | :heavy_check_mark:      | :heavy_check_mark: | :heavy_check_mark: |
| SAC      | :heavy_check_mark:      | :heavy_check_mark: | :x:                |
| TQC      | :heavy_check_mark:      | :heavy_check_mark: | :x:                |
| TD7      | :white_check_mark:(LAP) | :x:                | :x:                |

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
