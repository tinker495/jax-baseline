# JSON 네트워크 설정

YAML의 `actor_model`과 `critic_model`에서 각각 JSON 파일을 지정합니다.
Q-Net 계열은 `model`을 사용합니다.

```yaml
base:
  actor_model: models/mlp_512x3_relu.json
  critic_model: models/mlp_256x2_relu.json
```

MLP의 `layers`는 순서대로 은닉층의 너비를 나타냅니다. `[512, 512, 512]`는
512 유닛 은닉층 세 개입니다. 입력 크기와 출력 헤드는 환경과 알고리즘에서 결정합니다.
`activation`에 문자열을 지정하면 모든 은닉층에 적용하고, 배열을 지정하면 각 층에
차례로 적용합니다. 배열 길이는 `layers`와 같아야 합니다.

```json
{
  "type": "mlp",
  "layers": [512, 256, 128],
  "activation": ["relu", "silu", "tanh"],
  "embedding_mode": "normal"
}
```

`layers: []`는 은닉층을 생략합니다. 모델 파일을 지정하지 않으면 CLI의
기본 네트워크를 사용합니다.
지원하는 활성화 함수는 `relu`, `tanh`, `elu`, `gelu`, `silu`, `sigmoid`,
`leaky_relu`, `identity`입니다. 너비는 양의 정수여야 하며 알 수 없는 키나
활성화 함수는 오류로 처리합니다.

TD7의 MLP에는 은닉층이 하나 이상 필요합니다. 첫 층의 너비는 해당 역할의
SALE 인코더와 입력 투영 너비에도 적용됩니다. SIMBAv2 TD7의 `blocks`는
인코더와 네트워크의 각 잔차 블록 구간에 동일하게 적용됩니다.

SIMBA, SIMBAv2, FlashSAC의 잔차 네트워크는 블록 너비와 개수를 지정합니다.
Flax의 DDPG, TD3, SAC, TQC, CrossQ, TD7에서 MLP, SIMBA, SIMBAv2는
JSON의 `type`으로 선택하며, actor와 critic에 서로 다른 구조를 사용할 수 있습니다.
XQC와 Haiku 빌더는 MLP를 사용합니다.

```json
{
  "type": "simba",
  "width": 512,
  "blocks": 2,
  "activation": "relu",
  "embedding_mode": "normal"
}
```

`type`에는 `simba`, `simbav2`, `flashsac`을 사용할 수 있습니다.
`flashsac`은 FlashSAC 알고리즘의 블록입니다.

관측값의 평균과 표준편차를 누적해 정규화하는 기능은 네트워크 구조와 독립적입니다.
`dpg`와 `pg`에서는 CLI의 `--obs_rms_norm` 또는 YAML의 `obs_rms_norm: true`로
설정하며 기본값은 꺼짐입니다. FlashSAC은 외부 관측 RMS 정규화를 지원하지 않습니다.
`blocks: 0`은 잔차 블록을 생략하며 입력 임베딩과 출력 헤드는 유지합니다.

## 이미지 임베딩

이미지 관측의 인코더는 각 모델 JSON의 `embedding_mode`로 지정합니다.
기본값은 `normal`이며, Flax는 `normal`과 `resnet`을 지원하고 Haiku는
`normal`을 지원합니다. FlashSAC은 벡터 관측과 `normal` 설정만 허용합니다.
벡터 관측에는 이미지 인코더를 적용하지 않습니다.

```json
{
  "type": "mlp",
  "layers": [512, 256],
  "activation": "relu",
  "embedding_mode": "resnet"
}
```

`actor_model`의 설정은 `actor_` 및 공유 `unified_` 이미지에 적용됩니다.
`critic_model`의 설정은 critic 전용 `critic_` 이미지에 적용됩니다.
공유 이미지의 임베딩은 actor가 생성하며 critic도 같은 임베딩을 사용합니다.

## 경로와 입출력

`exp`는 모든 모델 상대 경로를 **YAML 파일이 있는 디렉터리** 기준으로 해석합니다.
`--set actor_model=...` 등의 덮어쓰기에도 같은 기준을 적용합니다.
`pg`, `dpg`, `qnet`, `impala`, `apex-dpg`, `apex-qnet`을 직접 실행할 때는
현재 디렉터리를 기준으로 해석합니다.

모델 JSON을 검증하고 펼쳐진 설정과 실행 명령을 확인합니다.

```bash
uv run exp experiments/configs/pg_mjlab_go1.yaml --dry-run
```

정규화된 JSON을 파일로 내보내려면 출력 디렉터리를 지정합니다.
아래 명령은 학습을 실행하지 않고 `runs/models/001-actor_model.json` 등의 파일을
작성하며, 출력된 실행 명령도 내보낸 파일을 참조합니다.

```bash
uv run exp experiments/configs/pg_mjlab_go1.yaml --dry-run --export-models runs/models
```

내보낸 파일은 수정한 후 모델 입력으로 다시 사용할 수 있습니다.
`--export-models`의 출력 디렉터리는 현재 디렉터리를 기준으로 해석합니다.
`--dry-run`을 생략하면 내보낸 모델을 사용해 학습도 실행합니다.

```bash
uv run exp experiments/configs/pg_mjlab_go1.yaml --dry-run --set actor_model=models/mlp_256x2_relu.json
```
