# haiku-baseline


## Implemented Environments

| **Name**                | **Q-Net based**              | **Actor-Critic based**       | **DDPG based**               | 
| ----------------------  | ---------------------------- | ---------------------------- | ---------------------------- |
| Gym                     | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |
| MultiworkerGym with Ray | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |
| Unity-ML Environments   | :heavy_check_mark:           | :heavy_check_mark:           | :heavy_check_mark:           |

## Implemented Algorithms

| **Name**            | **Refactored**               | ```Box```          | ```Discrete```     | ```Per```          | ```N-step```       | ```NoisyNet```     | ```Munchausen```   |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| DQN                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| C51                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QRDQN               | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| IQN                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| A2C                 | :heavy_check_mark: 			 | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| PPO                 | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| DDPG                | :heavy_check_mark:           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| SAC                 | :heavy_check_mark:           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| TD3                 | :heavy_check_mark:           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| TQC                 | :heavy_check_mark:           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |