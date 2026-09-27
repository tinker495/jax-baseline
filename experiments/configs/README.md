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
기본 로그 경로는 `runs/<category>/`이며, 예를 들어 Go1 PG 실행은 `runs/mjlab/PG/`에 저장한다.
`JAXBL_LOG_DIR`로 `runs`를 바꿀 수 있고, YAML 또는 `--set logdir=...`로 지정한 경로가 우선한다.
Aim의 experiment와 W&B의 project도 `category`를 사용하므로 같은 카테고리의 알고리즘을
함께 비교할 수 있다. 로컬 폴더의 `PG` 등은 `experiment_name`을 따른다. sweep 없이 runner를
직접 실행하면 Aim/W&B의 그룹 이름도 `experiment_name`을 사용한다.

| 폴더                                 | 실험 대상                          | 대표 설정                                                                                                                             |
| ------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| [atari_100k/](atari_100k/)           | Atari 100K 샘플 효율 실험          | [Breakout DQN 계열](atari_100k/dqn_breakout100k.yaml), [Asterix DQN 계열](atari_100k/dqn_asterix100k.yaml)                            |
| [atari/](atari/)                     | 일반 Atari 실험과 분산 학습        | [Breakout DQN](atari/dqn_breakout.yaml), [Breakout PG](atari/pg_breakout.yaml), [Space Invaders APE-X](atari/apex_spaceinvaders.yaml) |
| [mjlab/](mjlab/README.md)            | mjlab의 Cartpole·보행·로봇팔 실험  | [권장 설정과 출처](mjlab/README.md), [Go1 rough PPO](mjlab/pg_mjlab_go1_rough.yaml), [YAM PPO](mjlab/pg_mjlab_yam.yaml)               |
| [mujoco/](mujoco/)                   | Humanoid·Walker2d 연속 제어        | [Humanoid DPG](mujoco/dpg_humanoid.yaml), [Walker2d APE-X](mujoco/apex_dpg_walker2d.yaml), [Humanoid XQC](mujoco/xqc_humanoid.yaml)   |
| [box2d/](box2d/)                     | LunarLander·BipedalWalker          | [LunarLander PG](box2d/pg_lunarlander.yaml), [BipedalWalker IMPALA](box2d/impala_bipedal.yaml)                                        |
| [classic_control/](classic_control/) | 고전 제어 환경                     | [Acrobot DQN](classic_control/dqn_acrobot.yaml)                                                                                       |
| [models/](models/)                   | 여러 실험이 공유하는 네트워크 JSON | [모델 스키마](models/README.md)                                                                                                       |

`atari_100k/`에는 환경 상호작용 100,000회를 목표로 하는 설정을 둔다.
`steps`의 의미는 실행 계열에 따라 다르므로 값만 보고 분류하지 않는다.
예를 들어 `atari/impala_breakout_ppo.yaml`의 `steps: 1e5`는 learner 학습 횟수이다.

## 학습 효율을 우선하는 YAML 설계

이 디렉터리의 학습용 YAML은 **실험 목적과 학습 품질을 유지하면서, 지원되는 실행 경로에서
학습 효율을 최대화하는 기본 설정**으로 설계해야 한다. 충분한 로그 간격, 벌크 학습,
체크포인팅을 적극 활용해 반복적인 dispatch·CPU↔GPU 동기화·파일 I/O 비용을 줄인다.
새 설정을 추가하거나 기존 설정을 수정할 때 다음 기준을 적용한다.

- **로그 간격**: `log_interval`을 명시하고 worker 수와 실행 계열의 step 단위에 맞춘다.
  일반 `dpg`·`qnet`의 vector rollout에서는 전체 worker의 transition 기준이다.
  예를 들어 mjlab의 `worker: 1024`, `log_interval: 102400`은 100 vector steps마다 기록한다.
  이 환경에서 `1000` 같은 작은 기본값은 매 vector step마다 로깅을 유발한다.
  필요한 관측 해상도를 유지하는 충분한 간격을 선택해 상세 진단과 host 전송이
  매번 발생하지 않도록 한다. 디버깅용 짧은 간격은 실행 override로 지정한다.
- **벌크 학습**: 지원하는 알고리즘에서는 `train_freq`, `gradient_steps`,
  `max_bulk_updates_per_pulse`를 함께 설계해 여러 update가 compiled pulse로 실행되게 한다.
  환경 상호작용 대비 update 수, actor·target 갱신 주기와 메모리 한도를 확인한다.
  update 비율이 같아도 학습을 모아서 실행하면 행동 정책이 갱신되는 시점은 달라질 수 있다.
  따라서 chunk 크기를 무조건 키우지 않고 처리량·메모리·학습 성능으로 결정한다.
- **체크포인팅**: 지원하는 `dpg`·`qnet` 설정에서는 `use_checkpointing: true`로
  episode 단위 수집과 학습 pulse를 묶는 경로를 우선 검토한다.
  여기서 체크포인팅은 TD7식 행동 정책 snapshot과 학습 스케줄이며, 모델 파일 저장 주기와는
  별개다. 학습 시점과 행동 정책에 영향을 주므로 알고리즘·환경·비교 실험 조건에 맞춰 선택한다.
  비활성화가 필요한 경우 YAML 주석에 이유를 남긴다.
- **전송·평가 비용**: 환경과 replay를 가능한 한 소비 장치에 유지하도록 backend를 선택한다.
  `eval_num`·`eval_eps`와 저장·진단 빈도는 실험 목적에 필요한 수준으로 정한다.
  학습·평가 예산이나 평가 의미를 바꿔 얻은 시간 감소는 실행 최적화와 구분해 기록한다.
- **측정 근거**: 컴파일·워밍업을 제외한 처리량과 시작·평가·저장을 포함한 전체 시간을
  구분하고, 동일 환경·worker 수·update 예산에서 비교한다. 학습 곡선과 목표 성능 도달 시간도
  함께 확인한다. 단일 환경의 최적값을 모든 YAML에 일괄 적용하지 않는다.

현재 mjlab의 [FlashSAC G1 설정](mjlab/flashsac_mjlab_g1.yaml)은 충분한 로그 간격과 GPU replay,
vector step당 두 update를 사용한다. [MuJoCo FlashSAC 설정](mujoco/flashsac_humanoid.yaml)은
32-step 수집 후 32-update 벌크 학습을 사용하며, 정책 갱신 지연을 주석으로 명시한다.
이 값들은 해당 실험의 예시이며 모든 환경에 대한 최적값을 보장하지 않는다.

상속되는 `base`와 variant override를 포함한 최종 실행 인자를 `--dry-run`으로 확인한다.
처리량·학습 성능 측정 기록은 Git에서 제외된 `runs/`에 보관한다.
작성·변경 시 준수할 규칙은 [AGENTS.md](AGENTS.md)에 정리한다.

## 실행과 기록

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

각 실행 폴더의 `run.json`에는 기본값과 override가 반영된 인자, 실제 모델 JSON,
학습·평가 환경 backend와 seed, 평가 규칙, Git hash와 미커밋 diff, 설치된 패키지와
`uv.lock`, JAX 장치 정보가 저장된다. sweep에서는 원본 YAML과 활성 variant 번호도
함께 저장한다. TensorBoard·Aim·W&B 모두 같은 형식이며, 기록 저장에 실패하면 학습을
시작하지 않는다. Git 저장소나 lockfile이 없는 설치 환경에서는 해당 정보의 부재를 명시한다.

일반 `qnet`·`dpg`·`pg` 실행에서는 `eval_eps`로 평가 한 번의 episode 수를 지정한다
(기본 20). `eval_num`은 전체 학습 중 주기적 평가의 목표 횟수이며, 정상 종료 시 최종
평가는 별도로 수행한다. 분산 계열의 `rollout/` 지표는 학습 중 행동 정책의 결과이며,
별도의 frozen-policy 평가로 해석하지 않는다.

| 지표                    | 의미                                                                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| `progress/env_steps`    | 평가와 자동 reset용 dummy 행을 제외한 실제 학습 transition 수                                                   |
| `progress/update_steps` | 수행된 minibatch 업데이트 횟수. 같은 minibatch의 actor·critic 갱신은 한 번으로 센다.                            |
| `time/elapsed_seconds`  | 학습 시작 이후 경과 시간. 학습 중 평가·worker 준비와 정리는 포함하고, 모델 초기화·체크포인트 직렬화는 제외한다. |

기존 그래프의 step 축은 유지한다. 일반 계열의 `steps` 예산은 vector worker의 slot 수로
계산하며 dummy 행을 포함할 수 있다. 분산 계열의 예산은 learner 반복 횟수다.
계열 간 비교에는 위 지표 중 목적에 맞는 단위를 사용한다.

`--dry-run`은 활성 variant 전체의 옵션 이름·값·알고리즘 및 모델 backend 조합·모델 JSON을
검사한다. 오류가 있으면 variant 실행과 모델 export 전에 실패한다. 실제 환경 생성이나
장치 할당은 하지 않으므로 환경 설치 상태·관측/행동 공간·GPU 가용성은 실행 시 확인한다.
