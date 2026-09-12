# Loss 지표 정의

로컬 `pg`, `dpg`, `qnet`의 학습 지표다. 기존 logger와 기록 주기를 사용하며,
`loss/`에는 목적함수와 진단 통계가 함께 들어간다. Optimizer 지표는 `optim/`에 기록한다.
지표가 작아지거나 Q값이 커졌다는 사실만으로 정책 성능 개선을 판단하지 않는다.

## PG

A2C·PPO·SPO·TPPO에 적용한다. 표의 키는 `loss/` 접두사를 생략했다.

| 키                                                     | 정의와 측정 시점                                                                                                          |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `critic_loss`, `actor_loss`, `entropy_loss`            | 기존 손실 정의를 유지한다. `entropy_loss`는 음의 entropy다.                                                               |
| `actor_objective`                                      | 실제 미분한 actor 목적함수. Entropy advantage shaping을 켜면 additive entropy 항을 다시 더하지 않는다.                    |
| `explained_variance`                                   | 업데이트 전 rollout의 `1 - Var(target - value) / Var(target)`. Target 분산이 0이면 `NaN`이다.                             |
| `value_mean`, `value_std`, `mean_target`, `target_std` | 업데이트 전 rollout의 value와 return 통계.                                                                                |
| `advantage_mean`, `advantage_std`                      | 정규화와 entropy advantage shaping 전 advantage 통계.                                                                     |
| `approx_kl`                                            | PPO·SPO에서 `l = log_pi_new - log_pi_old`일 때 `mean(exp(l) - 1 - l)`. Old policy 표본으로 `KL(old \|\| new)`를 추정한다. |
| `clip_fraction`                                        | PPO·SPO의 `mean(abs(exp(l) - 1) > ppo_eps)`. 실제 objective가 clipped branch를 선택한 비율과는 다르다.                    |
| `kl_divergence`                                        | TPPO의 기존 exact KL. PPO의 ratio clipping 통계로 대체하지 않는다.                                                        |
| `policy_std_mean/min/max`, `log_std_mean/min/max`      | 연속 정책 Gaussian 파라미터의 표준편차와 log 표준편차. Bounded action 표본의 표준편차는 아니다.                           |

Rollout 통계는 rollout마다 한 번 측정한다. Surrogate 목적함수, KL, clip fraction,
정책 scale, optimizer 지표는 각 minibatch의 업데이트 시점에서 계산한 뒤 epoch 전체를 평균한다.

## DPG

DDPG·TD3·SAC·TQC·CrossQ·XQC·FlashSAC·TD7에 적용한다.

| `loss/` 키                                                  | 정의와 적용 조건                                                                                                                                    |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qloss`, `critic1_loss`, `critic2_loss`                     | 실제 critic 목적함수와 critic별 항. 일반 twin critic은 합, FlashSAC는 두 CE의 평균이다. TQC는 각 critic의 quantile Huber를 기록한다.                |
| `targets`                                                   | 실제 critic target 평균. SAC·TQC에서 이전에 잘못 기록하던 `-actor_loss`를 교정했다.                                                                 |
| `q1_mean/std/min/max`, `q2_mean/std/min/max`                | 업데이트에 사용한 replay action의 critic 예측. TQC는 각 critic 분포의 평균, XQC·FlashSAC는 support 기댓값이다.                                      |
| `target_std/min/max`                                        | Replay 표본별 target 기댓값의 통계.                                                                                                                 |
| `actor_loss`, `actor_q_mean`                                | 실제 actor 목적함수와 그 목적함수에 사용한 Q값의 평균. Replay action Q와 별도다.                                                                    |
| `td_signed_mean`, `td_abs_mean`, `td_abs_p95`               | `target - prediction`을 모든 critic과 표본에 걸쳐 요약. `td1_*`, `td2_*`는 critic별 값이다.                                                         |
| `q_disagreement`                                            | Twin critic의 `mean(abs(q1 - q2))`. 실제 return 대비 과대평가량은 아니다.                                                                           |
| `entropy`, `policy_target_entropy`, `entropy_gap`           | SAC 계열의 `-mean(log_pi)`, sigma에서 계산한 목표 entropy, 두 값의 차이. 실제 actor 표본과 tanh 보정을 재사용한다.                                  |
| `actor_entropy_term`                                        | SAC 계열 actor 목적함수의 `mean(alpha * log_pi)` 항.                                                                                                |
| `ent_coef`, `ent_coef_loss`                                 | Alpha와 자동 temperature 목적함수 `mean(alpha * (entropy - target_entropy))`. 고정 alpha에서는 temperature loss와 optimizer 지표를 기록하지 않는다. |
| `policy_std_*`, `log_std_*`                                 | SAC 계열 Gaussian 파라미터의 mean/min/max.                                                                                                          |
| `criticN/quantile_spread`, `criticN/quantile_crossing_rate` | TQC 각 critic의 분위수 폭과 인접 분위수 역전율.                                                                                                     |

Actor와 temperature 지표는 실제 해당 optimizer가 갱신된 횟수로 평균한다.
갱신을 건너뛴 스텝은 관측값 0으로 섞지 않으며, 갱신이 전혀 없는 로그 구간에서는 키를 생략한다.
실제 갱신에서 나온 `NaN`은 유지한다. Chunk와 단일 update가 섞여도 지표별 관측 수를 합산한다.
TD7의 기존 encoder loss와 value bounds도 유지한다.

## QNET와 분포형 진단

DQN·C51·QRDQN·IQN·FQF·SPR·BBF 및 HL-Gauss 변형에 적용한다.

| `loss/` 키                                                                     | 정의와 적용 조건                                                                                                                                  |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q_mean/std/min/max`, `target_mean/std/min/max`                                | Replay의 선택 행동 Q와 target 기댓값. FQF는 학습된 fraction 구간 폭으로 가중한다.                                                                 |
| `td_signed_mean`, `td_abs_mean`, `td_abs_p95`                                  | Scalar 기댓값의 `target - prediction`. 기존 MSE, CE, quantile Huber와 별개다.                                                                     |
| `categorical_cross_entropy`, `categorical_kl`, `target_entropy`                | 같은 안정화된 log로 계산한 `CE`, `KL(target \|\| online)`, `H(target)`. `CE = KL + H` 관계를 만족한다. 기존 최적화 CE의 epsilon과는 다를 수 있다. |
| `online_edge_mass_low/high`, `target_edge_mass_low/high`                       | 현재 분포와 투영된 target의 첫/마지막 bin 확률.                                                                                                   |
| `support_clipped_mass_low/high`                                                | 투영하기 전 support 범위 밖의 target 확률 질량. C51은 action mixture와 atom 확률로 가중한다. HL-Gauss는 변환 전 scalar target의 범위 이탈률이다.  |
| `quantile_spread`                                                              | 선택 행동의 online 분위수 최대값과 최소값 차이. Epistemic uncertainty로 해석하지 않는다.                                                          |
| `quantile_crossing_rate`                                                       | Tau 순서로 정렬한 뒤 인접 Q 분위수가 역전된 비율. IQN의 무작위 tau 순서를 그대로 비교하지 않는다.                                                 |
| `fraction_entropy`, `fraction_width_min/max`                                   | FQF fraction 분포의 entropy와 구간 폭 범위. 기존 `fqf_loss`, `tau` histogram도 유지한다.                                                          |
| `total_loss`, `weighted_rprloss`                                               | SPR·BBF의 실제 `qloss + spr_weight * rprloss`와 가중 representation 항. Raw `rprloss`도 유지한다.                                                 |
| `online_action_churn`, `target_action_churn`, `online_target_action_agreement` | SPR·BBF의 학습 전후 greedy action 변화율과 학습 후 online/target 일치율.                                                                          |
| `action_churn_updates_spanned`                                                 | Churn 측정 전후 사이의 실제 optimizer update 수. Bulk chunk 크기에 따른 측정 구간 차이를 표시한다.                                                |

XQC·FlashSAC의 categorical CE/KL/entropy/edge 통계는 `loss/criticN/`에 기록하고,
support clipping은 `loss/`에 기록한다. 기존 `target_stds`는 target 분위수 내부 표준편차의
평균이며 새 `target_std`와 다르다. 기존 FQF `targets`의 평균 방식도 유지한다.

SPR·BBF의 action 지표는 로그가 필요한 마지막 update 또는 bulk chunk에서만 측정한다.
동일한 replay 관측 최대 32개와 고정 진단 키를 사용하고 학습용 RNG는 소비하지 않는다.
Logger가 없거나 로그 주기가 아니면 추가 forward를 실행하지 않는다. Churn은 해당 update 또는
chunk 전후의 변화이며 서로 다른 로그 시점의 action 변화율은 아니다.

## PER와 optimizer

PER 활성화 시 learner가 replay에 전달하는 priority의 `loss/priority_mean/max/p95`와
실제 loss에 적용한 weight의 `loss/is_weight_mean/min/max/ess`를 기록한다.
`ESS = sum(w)^2 / sum(w^2)`는 현재 배치의 weight 통계다. 버퍼 전체의 다양성을 뜻하지 않는다.
`loss/unweighted_loss`는 importance weight를 적용하기 전의 동일한 critic loss다.
TD7의 priority 경로는 importance weight를 사용하지 않으므로 weight 1을 기준으로 기록한다.
FlashSAC는 PER를 지원하지 않는다.

| `optim/{optimizer}_` 접미사 | 정의                                                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `grad_norm_pre_clip`        | Optimizer에 전달된 gradient pytree의 전체 L2 norm.                                                                |
| `grad_clip_fraction`        | 설정된 global norm clipping 임계값을 초과한 update의 비율. Optimizer 내부의 개별 변환 clipping은 포함하지 않는다. |
| `update_norm`               | Optimizer가 반환한 update의 L2 norm. 이후 parameter projection/reset은 포함하지 않는다.                           |
| `parameter_norm`            | 해당 optimizer update 직전 parameter pytree의 L2 norm.                                                            |
| `learning_rate`             | 해당 update에 공급한 scalar/schedule 값. Adaptive preconditioning까지 포함한 파라미터별 유효 step size는 아니다.  |

Optimizer 이름은 `actor`, `critic`, `ent_coef`, `q`, `fraction`, `actor_encoder`,
`critic_encoder` 중 해당하는 값이다. Schedule의 실제 update counter와 periodic optimizer
reset을 따른다. Experiment factory는 자동으로 계측한다. 직접 주입한 Optax factory도 기존처럼
사용할 수 있으며, 계측이 필요하면 `track_optimizer`로 감싼다. 내부 optimizer state는
`OptimizerMetricsState.inner_state`에 보관한다.

DPG·QNET bulk 로그의 std, min/max, p95는 **update별 통계를 평균한 값**이다.
여러 배치의 원소를 합쳐 다시 구한 std나 p95가 아니다. Parameter reduction과 percentile에는
계산 비용이 있으며 지표 추가가 학습 처리량을 유지한다고 보장하지 않는다.
