# Breakout-v4 in 100K step

## DQNs
![DQNs](figures/dqn_breakout_100k.png)

## Max Reward 
| **Algorithm** | Signed Reward | Original Reward |
| -------- | ----- | ----- |
| Rainbow(DQN)  | 0.65 | 5.75 |
| Rainbow  | 2.15 | 16.0 |
| DER(DQN) | 2.25 | 18.0 |
| DER      | 2.3 | 22.0 |
| SPR      | 4.75 | 33.0 |
| SR-SPR   | 5.95 | 53.0 |
| BBF      | **19.65** | **299.4** |

## Algorithm Specifications
- **Rainbow(DQN)**: Rainbow DQN without C51, replay ratio (rr) = 0.25
- **Rainbow**: Standard Rainbow DQN, rr = 0.25
- **DER**: Data Efficient Rainbow, rr = 2
- **DER(DQN)**: Data Efficient version of basic DQN
- Note: Simple Rainbow DQN was not scaled by the replay ratio factor.