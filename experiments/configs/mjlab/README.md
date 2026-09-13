# mjlab 실험 예시

저장소의 `mjlab>=1.6,<1.7` 의존성에 맞춰 **공식 v1.6.0** 환경과 PPO 설정을
조사했다. 아래 PPO 예시는 공식 RSL-RL 설정을 이 저장소의 JAX PPO 인자로 옮긴
시작점이다. 별도 PPO 구현을 사용하므로 공식 학습 결과의 재현을 보장하지 않는다.
기존 평지 보행에서 거친 지형으로 확장하거나, Cartpole과 YAM으로 제어 과제를
바꿔 비교할 수 있다.

| PPO 예시                             | 환경 ID                                            | 권장 용도                                                              | 공식 PPO 설정 출처                                                                                      |
| ------------------------------------ | -------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [Cartpole](pg_mjlab_cartpole.yaml)   | `Mjlab-Cartpole-Balance`, `Mjlab-Cartpole-Swingup` | 작은 모델로 균형 유지와 스윙업 비교; 두 환경을 순서대로 실행           | [Cartpole](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/cartpole/cartpole_env_cfg.py) |
| [Go1 rough](pg_mjlab_go1_rough.yaml) | `Mjlab-Velocity-Rough-Unitree-Go1`                 | 기존 [Go1 flat](pg_mjlab_go1.yaml) 이후 사족 보행의 지형 적응 비교     | [Go1](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/config/go1/rl_cfg.py)     |
| [G1 rough](pg_mjlab_g1_rough.yaml)   | `Mjlab-Velocity-Rough-Unitree-G1`                  | 기존 [G1 flat](pg_mjlab_g1.yaml) 이후 휴머노이드 보행의 지형 적응 비교 | [G1](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/config/g1/rl_cfg.py)       |
| [YAM lift](pg_mjlab_yam.yaml)        | `Mjlab-Lift-Cube-Yam`                              | 상태 관측으로 로봇팔의 큐브 들기 학습                                  | [YAM](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/manipulation/config/yam/rl_cfg.py) |

Cartpole은 연속 effort 행동과 부드러운 보상을 사용하는 mjlab 환경이다.
등록된 ID와 flat/rough의 공통 PPO 설정은 각각
[Cartpole 등록](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/cartpole/__init__.py),
[Go1 등록](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/config/go1/__init__.py),
[G1 등록](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/velocity/config/g1/__init__.py),
[YAM 등록](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/manipulation/config/yam/__init__.py)에서 확인할 수 있다.

PPO 공통 권장값은 Adam, 학습률 `0.001`, 목표 KL `0.01`에 따른 적응 학습률,
`gamma: 0.99`, `lamda: 0.95`, `epoch_num: 5`, rollout을 네 minibatch로 분할,
policy/value clip `0.2`, gradient norm 상한 `1.0`이다.
새 PPO 예시는 공식 기본값처럼 advantage를 rollout 전체에서 정규화하므로
`gae_normalize_scope: batch`를 사용한다. 기존 flat 예시의 `minibatch`와 직접
비교할 때는 `--set gae_normalize_scope=batch`로 조건을 맞춘다.
공식 Gaussian 초기 표준편차 `1.0`과 value loss 계수 `1.0`은 현재 `pg` CLI에
대응 옵션이 없으므로 로컬 PPO 기본 구현을 사용한다.
([공식 PPO 설정 계약](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/rl/config.py),
[로컬 PG 옵션](../../cli/pg.py))

| 예시          | actor/critic 은닉층·활성화 | 관측 RMS | entropy | `worker` | `batch` | `mini_batch` | 공식 반복 수 | `steps`       |
| ------------- | -------------------------- | -------- | ------- | -------- | ------- | ------------ | ------------ | ------------- |
| Cartpole 각각 | 64, 64 · ELU               | 꺼짐     | 0.01    | 1,024    | 32      | 8,192        | 500          | 16,384,000    |
| Go1 rough     | 512, 256, 128 · ELU        | 켜짐     | 0.01    | 4,096    | 24      | 24,576       | 10,000       | 983,040,000   |
| G1 rough      | 512, 256, 128 · ELU        | 켜짐     | 0.01    | 4,096    | 24      | 24,576       | 30,000       | 2,949,120,000 |
| YAM lift      | 512, 256, 128 · ELU        | 켜짐     | 0.005   | 4,096    | 24      | 24,576       | 5,000        | 491,520,000   |

`pg`의 `batch`는 환경 하나의 rollout 길이, `mini_batch`는 한 번의 업데이트에
쓰는 샘플 수다. 위 값은 `mini_batch = worker × batch / 4`,
`steps = worker × batch × 공식 반복 수`로 환산했다. Cartpole의 예산은 환경별로 적용된다.
4,096개 환경은 [공식 G1 실행 예제](https://github.com/mujocolab/mjlab/blob/v1.6.0/README.md#training-examples)를
참고했다. Go1/YAM의 4,096개와 Cartpole의 1,024개는 이번 예시의 병렬도 선택이며
GPU별 최적값은 아니다. 환경 기본 episode 길이와 curriculum은 유지하고,
`env_observation_key`를 지정하지 않아 actor/critic 관측을 각각 전달한다.

저장소 루트에서 Python 3.11–3.13과 NVIDIA GPU를 사용한다. 필요한 extra를 설치한
환경에서는 `--no-sync`로 해당 설치 구성을 유지한다.

```bash
uv sync --extra mjlab --extra cuda12
uv run --no-sync exp experiments/configs/mjlab/pg_mjlab_cartpole.yaml --dry-run
uv run --no-sync exp experiments/configs/mjlab/pg_mjlab_cartpole.yaml
uv run --no-sync exp experiments/configs/mjlab/pg_mjlab_yam.yaml
uv run --no-sync exp experiments/configs/mjlab/pg_mjlab_go1_rough.yaml
uv run --no-sync exp experiments/configs/mjlab/pg_mjlab_g1_rough.yaml
```

`--dry-run`에서 명령·인자·모델 JSON을 확인할 수 있다. 실제 환경 생성이나 학습
수렴은 검사하지 않는다. 로컬 로그만 필요하면 `--set logger=tensorboard`를 추가한다.
기본 로그 경로는 `runs/mjlab/`이며, 실행별 설정과 seed는 `run.json`에 저장된다.

GPU 메모리가 부족하면 환경 수와 minibatch 크기를 함께 줄인다. 아래 Go1 예시는
네 minibatch와 10,000회 반복을 유지한다.

```bash
uv run --no-sync exp experiments/configs/mjlab/pg_mjlab_go1_rough.yaml --set worker=1024 --set mini_batch=6144 --set steps=245760000
```

seed는 `0`으로 고정했다. 결과를 비교할 때는 동일한 환경·예산에서
`--set seed=1`, `--set seed=2`로 반복하고 평가 보상과 분산을 함께 확인한다.

같은 환경의 FlashSAC 예시도 제공한다. 기존 [Go1 flat](flashsac_mjlab_go1.yaml)·
[G1 flat](flashsac_mjlab_g1.yaml)과
[FlashSAC 공식 GPU 시뮬레이터 설정](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/scripts/run_isaaclab.sh)을
기준으로 옮긴 시작점이다. 공식 IsaacLab 벤치마크와 환경·관측 구성·버퍼 크기가
다르며, 이 mjlab 환경들에서 최적화된 권장값이라는 의미는 아니다.

| FlashSAC 예시                              | 환경                               | 환경별 `steps` | GPU `buffer_size` |
| ------------------------------------------ | ---------------------------------- | -------------- | ----------------- |
| [Cartpole](flashsac_mjlab_cartpole.yaml)   | Balance와 Swingup을 순서대로 실행  | 50,000,896     | 1,000,000         |
| [Go1 rough](flashsac_mjlab_go1_rough.yaml) | `Mjlab-Velocity-Rough-Unitree-Go1` | 50,000,896     | 1,000,000         |
| [G1 rough](flashsac_mjlab_g1_rough.yaml)   | `Mjlab-Velocity-Rough-Unitree-G1`  | 50,000,896     | 1,000,000         |
| [YAM lift](flashsac_mjlab_yam.yaml)        | `Mjlab-Lift-Cube-Yam`              | 50,000,896     | 1,000,000         |

FlashSAC 공통값은 `worker: 1024`, `train_freq: 1024`, `gradient_steps: 2`,
`batch: 2048`, `learning_starts: 100000`, `n_step: 3`, `gamma: 0.99`다.
학습 시작 이후 벡터 step마다 critic 업데이트 두 번, `actor_update_period: 2`로
actor 업데이트 한 번을 수행한다. Adam 학습률은 `0.0003`에서 `0.00015`까지
cosine 방식으로 감소한다. 기존 FlashSAC ReLU 잔차 모델(actor 128×2, critic 256×2),
`target_update_tau: 0.01`, `ent_coef: auto_0.01`, `sigma_target: 0.15`,
reward 정규화를 공유한다. 관측 RMS 정규화와 prioritized replay는 사용하지 않는다.

새 예시는 관측 차원이 큰 rough 환경도 고려해 공식 1,000만 replay를 100만으로
줄였다. GPU 여유에 따라 `--set buffer_size=5e6` 등으로 조정한다. worker를 바꿀 때는
`train_freq`도 같은 값으로 바꿔 벡터 step당 업데이트 비율을 유지한다.
PPO와 샘플 효율을 비교하려면 `--set steps=...`로 전체 transition 예산도 맞춘다.

```bash
uv run --no-sync exp experiments/configs/mjlab/flashsac_mjlab_cartpole.yaml --dry-run
uv run --no-sync exp experiments/configs/mjlab/flashsac_mjlab_cartpole.yaml
uv run --no-sync exp experiments/configs/mjlab/flashsac_mjlab_go1_rough.yaml
uv run --no-sync exp experiments/configs/mjlab/flashsac_mjlab_g1_rough.yaml
uv run --no-sync exp experiments/configs/mjlab/flashsac_mjlab_yam.yaml
```

추가 후보인 `Mjlab-Lift-Cube-Yam-Rgb`, `Mjlab-Lift-Cube-Yam-Depth`,
`Mjlab-Multi-Cube-Seg-Yam`은 영상과 상태를 결합하는 `SpatialSoftmaxCNNModel`을
공식적으로 사용한다. 현재 MLP 예시의 환경 ID만 바꾸는 방식으로는 이 구성을
재현할 수 없어 보류했다.
([영상 관측 설정](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/manipulation/config/yam/env_cfgs.py),
[영상 PPO 모델](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/tasks/manipulation/config/yam/rl_cfg.py))

`Mjlab-Tracking-Flat-Unitree-G1`과 `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation`은
mjlab 형식의 motion NPZ를 먼저 준비하고 전달해야 한다. 공식 `train`의
`--env.commands.motion.motion-file`/`--registry-name`에 대응하는 입력을 이 저장소의
환경 구성 경로에 마련한 뒤 추가할 후보로 남긴다.
([공식 motion 입력](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/scripts/train.py#L67-L92),
[motion 준비](https://github.com/mujocolab/mjlab/blob/v1.6.0/docs/source/training/motion_imitation.rst))
