# Haiku-Baseline

Haiku-Baseline is the same Reinforcement Learning implementation as any Baseline implemented through the JAX and Haiku libraries.

It's not compared to any Baseline yet, but it's two to three times faster than the Torch and Tensorflow works that we've implemented before.
Using JAX's JIT(Just In Time) compilation, we optimized a series of courses for learning and constructed them based on functions. This allows you to see how JAX's capabilities can be used effectively in various Reinforcement Learning implementations.

This implementation is configured to flexibly solve the commonly used gym and unity ml environment for testing algorithms in various complex environments.

## Implemented Environments

| **Name**                | **Q-Net based**              | **Actor-Critic based**       | **DDPG based**               | 
| ----------------------  | ---------------------------- | ---------------------------- | ---------------------------- |
| Gym                     | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |
| MultiworkerGym with Ray | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |
| Unity-ML Environments   | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |

## Implemented Algorithms

| **Name**            | **Complete**                 | ```Box```          | ```Discrete```     | ```Per```          | ```N-step```       | ```NoisyNet```     | ```Munchausen```   |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| DQN                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| C51                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QRDQN               | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| IQN                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| FQF                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| A2C                 | :heavy_check_mark: 			 | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| TRPO                | TODO           		         |                    |                    |                    |                    |                    |                    |
| PPO                 | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| ACER                | TODO           		         |                    |                    |                    |                    |                    |                    |
| DDPG                | :heavy_check_mark:           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| SAC                 | :heavy_check_mark:           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| TD3                 | :heavy_check_mark:           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| TQC                 | :heavy_check_mark:           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| Ape-X-Qnets         | TODO           		         |                    |                    |                    |                    |                    |                    |
| Ape-X-DDPG based    | TODO           		         |                    |                    |                    |                    |                    |                    |
| IMPALA              | TODO           		         |                    |                    |                    |                    |                    |                    |

## Test 

To test atari and Rainbow-DQN(or C51, QRQDN, IQN, FQF)

```
python test/run_qnet.py --algo DQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0000625 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 2e5 --n_step 3 --per --double --dueling --noisynet --munchausen
python test/run_qnet.py --algo C51 --env BreakoutNoFrameskip-v4 --learning_rate 0.0000625 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 2e5 --n_step 3 --per --double --dueling --noisynet --munchausen --max 10 --min -10
python test/run_qnet.py --algo QRDQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0000625 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 2e5 --n_step 3 --per --double --dueling --noisynet --munchausen --n_support 32
python test/run_qnet.py --algo IQN --env BreakoutNoFrameskip-v4 --learning_rate 0.0000625 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 2e5 --n_step 3 --per --double --dueling --noisynet --munchausen --n_support 32
python test/run_qnet.py --algo FQF --env BreakoutNoFrameskip-v4 --learning_rate 0.0000625 --steps 1e7 --batch 64 --target_update 2000 --node 512 --hidden_n 1 --final_eps 0.01 --learning_starts 1000 --gamma 0.995 --buffer_size 2e5 --n_step 3 --per --double --dueling --noisynet --munchausen --n_support 32
```