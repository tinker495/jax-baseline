# 실험 설정

실험 대상과 벤치마크를 기준으로 분류한다. 알고리즘, 환경 이름, 병렬 실행 구성은 파일명으로 구분한다.

YAML의 `category`는 실험 주제이며 폴더 이름과 맞춘다. `runner`는 실행할 CLI를 지정한다.
같은 PPO도 `runner: pg`이면 일반 학습, `runner: impala`이면 분산 학습으로 실행한다.

```yaml
category: atari_100k
runner: qnet
base:
  env: BreakoutNoFrameskip-v4
  steps: 1e5
  model: ../models/mlp_512x1_relu.json
variants:
  - algo: DQN
```

`exp`는 실행 명령 앞에 `category: atari_100k`처럼 주제를 표시한다.

| 폴더                                 | 실험 대상                          | 대표 설정                                                                                                                             |
| ------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| [atari_100k/](atari_100k/)           | Atari 100K 샘플 효율 실험          | [Breakout DQN 계열](atari_100k/dqn_breakout100k.yaml), [Asterix DQN 계열](atari_100k/dqn_asterix100k.yaml)                            |
| [atari/](atari/)                     | 일반 Atari 실험과 분산 학습        | [Breakout DQN](atari/dqn_breakout.yaml), [Breakout PG](atari/pg_breakout.yaml), [Space Invaders APE-X](atari/apex_spaceinvaders.yaml) |
| [mjlab/](mjlab/)                     | mjlab의 Unitree Go1·G1 로봇 실험   | [Go1 PG](mjlab/pg_mjlab_go1.yaml), [G1 FlashSAC](mjlab/flashsac_mjlab_g1.yaml)                                                        |
| [mujoco/](mujoco/)                   | Humanoid·Walker2d 연속 제어        | [Humanoid DPG](mujoco/dpg_humanoid.yaml), [Walker2d APE-X](mujoco/apex_dpg_walker2d.yaml), [Humanoid XQC](mujoco/xqc_humanoid.yaml)   |
| [box2d/](box2d/)                     | LunarLander·BipedalWalker          | [LunarLander PG](box2d/pg_lunarlander.yaml), [BipedalWalker IMPALA](box2d/impala_bipedal.yaml)                                        |
| [classic_control/](classic_control/) | 고전 제어 환경                     | [Acrobot DQN](classic_control/dqn_acrobot.yaml)                                                                                       |
| [models/](models/)                   | 여러 실험이 공유하는 네트워크 JSON | [모델 스키마](models/README.md)                                                                                                       |

`atari_100k/`에는 환경 상호작용 100,000회를 목표로 하는 설정을 둔다.
`steps`의 의미는 실행 계열에 따라 다르므로 값만 보고 분류하지 않는다.
예를 들어 `atari/impala_breakout_ppo.yaml`의 `steps: 1e5`는 learner 학습 횟수이다.

저장소 루트에서 실행한다. `--dry-run`은 학습을 시작하지 않고 실행 명령과 모델 정의를 확인한다.

```bash
uv run exp experiments/configs/atari_100k/dqn_breakout100k.yaml --dry-run
uv run exp experiments/configs/atari/dqn_breakout.yaml
uv run exp experiments/configs/mjlab/pg_mjlab_go1.yaml
```

`model`, `actor_model`, `critic_model` 경로는 해당 YAML을 기준으로 해석한다.
각 실험 폴더에서는 공통 모델을 `../models/<name>.json`으로 참조한다.
`--set`으로 모델을 바꿀 때도 같은 상대 경로 규칙을 사용한다.

```bash
uv run exp experiments/configs/atari/dqn_breakout.yaml --dry-run --set model=../models/mlp_512x2_relu.json
```
