# Haiku-Baseline

Haiku-Baseline is the same Reinforcement Learning implementation as any Baseline implemented through the JAX and Haiku libraries.

It's not compared to any Baseline yet, but it's two to three times faster than the Torch and Tensorflow works that we've implemented before.
Using JAX's JIT(Just In Time) compilation, we optimized a series of courses for learning and constructed them based on functions. This allows you to see how JAX's capabilities can be used effectively in various Reinforcement Learning implementations.

This implementation is configured to flexibly solve the commonly used gym and unity ml environment for testing algorithms in various complex environments.

## Installation
```
pip install -r requirement.txt
pip install .
```
## Implemented Environments

| **Name**                | **Q-Net based**              | **Actor-Critic based**       | **DPG based**                |
| ----------------------  | ---------------------------- | ---------------------------- | ---------------------------- |
| Gymnasium               | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |
| MultiworkerGym with Ray | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |
| Unity-ML Environments   | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |

## Implemented Algorithms

### Q-Net bases

| **Name**            | **Complete**                 | ```Per```          | ```N-step```       | ```NoisyNet```     | ```Munchausen```   |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| DQN                 | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| C51                 | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QRDQN               | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| IQN                 | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| FQF                 | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Ape-X-Qnets         | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### Actor-Critic based

| **Name**            | **Complete**                 | ```Box```          | ```Discrete```     | ```LSTM```         |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ |
| A2C                 | :heavy_check_mark: 			 | :heavy_check_mark: | :heavy_check_mark: | TODO               |
| PPO                 | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | TODO               |
| Truly PPO           | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | TODO               |
| IMPALA              | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | TODO               |
| IMPALA-PPO          | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | TODO               |

### DPG bases

| **Name**            | **Complete**                 | ```Per```          | ```N-step```       |
| ------------------- | ---------------------------- | ------------------ | ------------------ |
| DDPG                | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: |
| TD3                 | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: |
| SAC                 | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: |
| TQC                 | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: |
| Ape-X-DPGs    	  | :heavy_check_mark:           |:heavy_check_mark:  | :heavy_check_mark: |

## Test 

To test atari with DQN(or C51, QRQDN, IQN, FQF)
```
python run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0002 \
		--steps 5e5 --batch 32 --train_freq 1 --target_update 1000 --node 512 \
		--hidden_n 1 --final_eps 0.01 --learning_starts 20000 --gamma 0.995 --clip_rewards
```

Only 15 minutes is sufficient to run 50K steps on DQNs learning atari breakout (540 steps/sec). 
This performance measurement was on Nvidia RTX3080 and AMD Ryzen 9 5950X in a single process.
```
score : 9.600, epsilon : 0.010, loss : 0.181 |: 100%|███████| 500000/500000 [15:24<00:00, 540.88it/s]
```