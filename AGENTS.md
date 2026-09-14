## Code Review Rules

사람/AI 리뷰어 공통 규칙이다. 포맷·린트 등 기계적 검사는 CI 소관이므로 리뷰에서 다루지 않는다.

아래의 테스트 수정·삭제 제안과 co-migration 요구는 리뷰 판단 기준이며,
에이전트에게 테스트 변경 권한을 부여하지 않는다.
실행 권한은 이 문서의 "테스트의 소유권과 변경 권한"을 따른다.

### 리뷰 언어와 용어

- 리뷰 코멘트와 findings 본문은 한국어와 영어를 병기로 작성한다. 코드 식별자와 아키텍처 용어는 영어 원형을 유지한다.

### 과감한 아키텍처 변경 제안

- diff 라인의 최소 수정 제안에 머무르지 않는다. 변경이 구조 문제(개념의 중복 소유, shotgun surgery, 위임만 하는 wrapper, 스테이지 경계 침식)를 드러내면 모듈 이동·병합·삭제, 소유권 재배치 같은 더 깊은 구조 변경을 명시적 대안으로 제안한다.
- 판단 기준은 "최소 변경"이 아니라 "최소 컨텍스트"다: 유지보수자가 알아야 할 개념 수를 줄이는 방향(좁은 인터페이스 + 깊은 구현의 deep module)을 우선하고, 순증보다 삭제로 푸는 diff를 높게 평가한다.
- speculative generality(단일 구현 인터페이스, 아무도 설정하지 않는 config, hand-rolled stdlib 재구현)는 발견 즉시 삭제 후보로 지적한다.
- 구조 제안에는 현재 구조가 유발하는 실제 비용을 근거로 붙이고, 심각도 순으로 제시한다.

## 테스트의 소유권과 변경 권한

### 기본 원칙

테스트는 구현의 부속물이 아니라 사용자가 소유하는 실행 가능한 제품 명세다.
에이전트는 제품을 구현하는 주체이지, 무엇을 올바른 제품으로 인정할지 결정하는 주체가 아니다.

에이전트는 테스트를 읽고 실행할 수 있지만, 작성·수정·삭제할 수 없다.
"기능 구현", "버그 수정", "리팩터링", "테스트 통과" 등의 작업 지시는
테스트 또는 합격 기준의 변경 권한을 부여하지 않는다.

모든 테스트는 작성자(에이전트가 아닌 사용자.)를 명시하고. 주석으로 해당 테스트 추가의 맥락을 상세한 설명으로 표기해야 한다.

### 보호 대상

기본적으로 저장소에 포함되는 모든 테스트와 제품의 합격 여부를 결정하는
검증 자산을 보호 대상으로 취급한다. 경로나 명칭에 관계없이 다음을 포함한다.

- 테스트 코드, assertion, 테스트 케이스와 매개변수.
- 기대값, fixture, snapshot, golden data 및 참조 결과.
- 독립 검증기·oracle, 평가 데이터, 합격 임계값과 허용 오차.
- 테스트 수집·실행·제외 설정과 CI의 검증 단계.

에이전트가 임의로 "내부 단위 테스트", "회귀 테스트", "테스트 보완" 등으로
분류하여 보호 대상에서 제외해서는 안 된다.

### 허용되는 작업

- 기존 테스트와 검증 자산을 읽고 실행하여 요구사항과 실패 원인을 분석한다.
- 사용자 요구사항을 만족하도록 제품 구현을 수정한다.
- 디버깅을 위한 임시 재현 코드, 탐색 스크립트, 관측 로그를 작성한다.
- 필요한 테스트 케이스, 반례, 명세의 모호함을 사용자에게 텍스트로 제안한다.

임시 검증 코드는 정식 테스트 디렉터리 밖의 임시 공간에만 작성한다.
커밋하거나 CI에 등록하거나 정식 테스트로 편입해서는 안 된다.
임시 검증의 성공을 사용자 승인 또는 제품 인수 기준 충족으로 간주하지 않는다.

### 금지되는 작업

- 테스트의 신규 작성, 수정, 삭제, 이동 또는 자동 재생성.
- 현재 구현의 출력을 정답으로 간주하여 기대값이나 snapshot을 확정하는 행위.
- 테스트를 통과시키기 위한 assertion 약화, 허용 오차 확대, 사례 제거,
  skip·xfail 처리, fixture 변경 또는 검증 단계 우회.
- 테스트 환경이나 특정 테스트 입력을 감지하여 정답을 하드코딩하거나,
  실패·위반·누락을 숨기는 방식으로 통과시키는 구현.
- 작업 편의를 위해 이 정책 또는 관련 보호 설정을 완화하는 행위.

### 테스트 실패와 명세 충돌 처리

테스트가 실패하면 먼저 제품 구현의 결함을 조사한다.
실패했다는 이유만으로 테스트가 잘못되었다고 판단하지 않는다.

반대로, 기존 테스트가 존재한다는 사실만으로 사용자에게 승인된
올바른 제품 명세라고 단정하지도 않는다.

테스트와 사용자 요구사항이 충돌하거나 기대 동작을 확정할 근거가 부족하면,
테스트를 변경하거나 잘못된 동작을 구현에 고착시키지 않는다.
충돌하는 요구사항, 재현 입력, 실제 결과, 필요한 사용자 결정을 보고하고
해당 판단에 의존하는 변경만 보류한다.

새 테스트가 필요해도 직접 추가하지 않는다.
입력 조건, 검증할 동작, 기대 결과의 근거를 사용자에게 제안한다.

### 완료 보고

실제로 실행한 테스트의 범위와 결과, 실행하지 못한 검증,
남은 실패 및 사용자 판단이 필요한 사항을 구분하여 보고한다.

## Graft — repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).

<!-- graft:end -->
