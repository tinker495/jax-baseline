# PR CI

모든 PR에서 `CI` 워크플로가 실행된다. Python 3.11은 `.python-version`과 맞추고,
uv 0.12.23과 `uv.lock`을 사용한다. 수동 실행도 지원한다.

| 필수 status check 이름 | 검사 범위                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| `Lint and format`      | 기존 pre-commit 전체 파일 검사: YAML, 파일 끝/공백, flake8, Black, isort, mdformat       |
| `CPU validation`       | core 의존성 설치·호환성, Python 문법, JAX CPU 초기화, CLI 도움말, 기존 실험 설정 dry-run |

첫 PR에서 두 작업의 성공을 확인한 뒤 main ruleset의 required status checks에 위 이름을
등록한다. 제공 앱은 **GitHub Actions**를 선택한다. `CI`는 워크플로 이름이며 required
check의 이름은 각 작업 이름이다. 이 PR 자체는 main ruleset을 변경하지 않는다.

## 기존 검증 자산 활용

`Lint and format`은 `uv sync --locked --only-group dev`로 개발 도구만 설치하고,
기존 `.pre-commit-config.yaml`의 모든 hook을 전체 파일에 적용한다. hook이 파일을
수정하면 검사는 실패하며 diff를 표시한다. 수정 내용을 자동 커밋하지 않는다.

`CPU validation`은 `uv sync --locked --no-default-groups`로 core 패키지만 설치한다.
CUDA, mjlab, Atari ROM, MuJoCo, Ray 등의 optional extra를 설치하지 않는다.
`JAX_PLATFORMS=cpu`로 장치를 제한하고 다음 기존 실행 경로를 확인한다.

- `uv pip check` 및 저장소 Python 소스의 `compileall`
- JAX CPU backend 초기화
- `qnet`, `dpg`, `pg`, `impala`, `apex-qnet`, `apex-dpg`, `exp`, `dashboard`의 `--help`
- `experiments/configs/*/*.yaml` 전체에 대한 `exp --dry-run`

dry-run은 기존 CLI의 옵션·알고리즘/backend 조합·모델 JSON 검증을 실행한다.
mjlab 등 GPU 실험 설정도 **설정만** 검사하며 환경 생성, 학습, 평가를 시작하지 않는다.
이 검사는 알고리즘 정확성이나 학습 성능을 검증하는 제품 테스트가 아니다.
현재 저장소에는 pytest 테스트 파일과 Pyrefly 설정이 없으므로 해당 작업을 만들지 않는다.
새 제품 테스트, 기대값, 허용 오차 또는 실험 설정도 추가·수정하지 않는다.

## 로컬 재현

저장소 루트에서 실행한다. 두 `uv sync` 명령은 각 CI 작업의 독립 환경을 재현하기 위해
설치 패키지를 교체한다.

```bash
uv sync --locked --only-group dev
uv run --no-sync pre-commit run --all-files --show-diff-on-failure

uv sync --locked --no-default-groups
export JAX_PLATFORMS=cpu
uv pip check
uv run --no-sync python -m compileall -q jax_baselines env_builder experiments model_builder replay_memory docs
uv run --no-sync python -c 'import jax; print(jax.devices())'
for command in qnet dpg pg impala apex-qnet apex-dpg exp dashboard; do
  uv run --no-sync "$command" --help || exit
done
for config in experiments/configs/*/*.yaml; do
  uv run --no-sync exp "$config" --dry-run || exit
done
```

## 외부 PR과 의존성 보안

- `pull_request` 이벤트와 GitHub 호스팅 러너를 사용한다. 기본 checkout은 PR merge
  commit을 검사한다. `pull_request_target` 또는 self-hosted runner를 사용하지 않는다.
- `GITHUB_TOKEN`은 `contents: read`만 허용하며 secrets를 전달하지 않는다.
  Checkout의 `persist-credentials: false`로 git 인증 정보를 작업 디렉터리에 남기지 않는다.
- 각 작업은 15분으로 제한하고 같은 PR의 이전 실행은 취소한다. setup-uv cache는 끈다.
- Checkout과 setup-uv는 공식 upstream에서 확인한 40자리 commit SHA로 고정한다.
  `dependabot.yml`은 GitHub Actions 업데이트 PR을 매주 확인한다. uv 실행 파일 버전은
  별도 고정값이므로 갱신 시 위 명령을 다시 실행한다.
- SHA 고정은 action 코드에 적용된다. 기존 pre-commit hook의 tag와 별도 빌드 의존성까지
  불변으로 만드는 것은 아니다. 기존 CODEOWNERS 규칙이 CI 파일 변경에도 적용된다.
