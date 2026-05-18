# da-template

[![Use this template](https://img.shields.io/badge/Use%20this%20template-2ea44f?style=for-the-badge&logo=github)](https://github.com/JungYeoni/da-template/generate)

개인 또는 소규모 팀을 위한 데이터분석·ML 프로젝트 템플릿입니다.  
새 프로젝트를 시작할 때 위 버튼을 눌러 레포를 생성하면, 일관된 디렉터리 구조·재현성 기준·협업 규칙·GitHub 자동화를 바로 사용할 수 있습니다.

---

## 적합한 프로젝트 유형

- 정형 데이터 EDA 및 분류/회귀
- 시계열 분석 및 예측
- 회귀/인과추론 (OLS, DiD, 패널 데이터)
- GIS 결합형 데이터 분석
- 논문·보고서용 시각화 + 인터랙티브 대시보드

---

## 디렉터리 구조

```
da-template/
├── CLAUDE.md                     # 프로젝트 분석 원칙 (Claude Code 지침)
├── pyproject.toml                # Python 의존성 및 도구 설정
├── requirements.txt              # 핵심 의존성 목록
├── .gitignore
│
├── .claude/
│   ├── CLAUDE.md                 # 전역 역할·스택 설정
│   ├── settings.json             # 민감 파일 접근 제한
│   ├── agents/                   # 서브에이전트 역할 정의
│   │   ├── data-scientist.md
│   │   ├── data-visualization.md
│   │   └── feature-engineer.md
│   ├── commands/                 # 슬래시 커맨드 (/timeseries 등)
│   └── rules/                    # 분석 규칙 (코드 스타일, 워크플로우 등)
│
├── .github/
│   ├── CODEOWNERS
│   ├── pull_request_template.md
│   ├── ISSUE_TEMPLATE/
│   │   ├── experiment.yml        # 실험 계획 이슈 템플릿
│   │   ├── bug_report.yml
│   │   └── config.yml
│   └── workflows/
│       ├── ci.yml                # lint + test
│       ├── notebook-smoke-test.yml
│       └── pr-title-lint.yml
│
├── configs/
│   ├── base.yaml                 # 공통 설정 (seed, split 비율, 경로 등)
│   ├── dev.yaml                  # 개발 환경 오버라이드
│   └── prod.yaml                 # 최종 제출 환경 오버라이드
│
├── data/
│   ├── raw/          # 원본 데이터 (git 추적 제외)
│   ├── interim/      # 중간 처리 결과
│   └── processed/    # 모델 입력용 최종 데이터
│
├── notebooks/        # 탐색·실험용 Jupyter 노트북
├── reports/          # 최종 보고서·시각화 산출물
│
├── src/
│   ├── features/
│   │   └── build_features.py    # 시계열·테이블·GIS 피처 함수
│   ├── modeling/
│   │   └── train.py             # 모델 학습 유틸리티
│   ├── evaluation/
│   │   └── evaluate.py          # 평가 지표 계산
│   └── visualization/
│       └── plots.py             # 정적 시각화 함수
│
└── tests/
    └── test_features.py         # 피처 단위 테스트
```

---

## 시작 방법

```bash
# 1. 저장소 클론
git clone https://github.com/JungYeoni/da-template.git my-project
cd my-project

# 2. 가상환경 생성 및 의존성 설치
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. 테스트 실행 (환경 확인)
pytest tests/ -v
```

---

## 새 실험을 시작하는 기본 흐름

1. **GitHub Issue 생성** — `[Experiment]` 템플릿으로 목표·데이터·분할 전략 문서화
2. **브랜치 생성** — `git checkout -b experiment/short-description`
3. **`configs/base.yaml` 확인** — random seed, split 비율, 경로 설정
4. **노트북 작성** — `notebooks/` 아래에서 EDA 및 실험
5. **재사용 함수 정리** — `src/` 하위 모듈로 이동 후 테스트 작성
6. **PR 생성** — `[Experiment]` 접두사, PR 체크리스트 작성

### PR 제목 규칙

| 접두사 | 사용 시점 |
|--------|----------|
| `[Experiment]` | 새 분석 실험 |
| `[Feature]` | 피처 추가·수정 |
| `[Fix]` | 버그 수정 |
| `[Docs]` | 문서 변경 |
| `[Refactor]` | 기능 변경 없는 코드 정리 |
| `[Chore]` | 의존성, 설정 변경 |

---

## 핵심 원칙

### 재현성
- `np.random.seed(42)` 고정
- 데이터 분할 기준을 코드 주석으로 명시
- 전처리는 `sklearn.Pipeline` 내에서만 수행

### 데이터 누수 방지
- 피처 생성 전에 train/val/test 분리 확정
- 시계열 rolling/lag은 `shift(1)` 후 계산
- 인코더·스케일러 파라미터는 학습 데이터에서만 fit

### 방법론적 타당성 우선
- 성능보다 통계적 가정 검증, 계수 해석, 한계점 명시를 우선
- 복잡한 모델보다 해석 가능하고 재현 가능한 방법 선택

---

## Claude Code 연동

`CLAUDE.md` (프로젝트 루트)와 `.claude/` 폴더가 Claude Code에서 자동으로 로드됩니다.

### 슬래시 커맨드

| 커맨드 | 기능 |
|--------|------|
| `/timeseries` | 시계열 분석 — ADF, ARIMA, Prophet |
| `/tabular` | 테이블 EDA — 기술통계, 이상치, 상관관계 |
| `/gis` | 지리공간 분석 — geopandas, folium |
| `/regression` | 회귀 분석 — OLS, VIF, 잔차 진단 |
| `/ml` | ML 파이프라인 — 전처리, 모델 비교, SHAP |
| `/visualization` | 시각화 — matplotlib, seaborn, plotly |

### 서브에이전트

`.claude/agents/` 파일을 Claude Code가 에이전트로 로드합니다.

| 에이전트 | 역할 |
|----------|------|
| `data-scientist` | 통계 분석, 시계열, 회귀/인과추론, sklearn ML |
| `feature-engineer` | 피처 설계·검증 (시계열, 테이블, GIS) |
| `data-visualization` | 정적 이미지(dpi=300) + Plotly/Streamlit 대시보드 |

> Claude Code 외 다른 코딩 에이전트(Cursor, Copilot 등)를 사용하는 경우에도 `CLAUDE.md`와 `src/` 코드가 분석 원칙 참고 문서로 활용될 수 있습니다.

---

## GitHub Actions

| 워크플로우 | 트리거 | 내용 |
|-----------|--------|------|
| `ci.yml` | push/PR → main | ruff lint, black format check, pytest |
| `notebook-smoke-test.yml` | PR에서 notebooks/ 변경 | 변경된 노트북 실행 가능 여부 확인 |
| `pr-title-lint.yml` | PR 생성/수정 | 제목 접두사 형식 확인 |
| `changelog.yml` | push → main | `CHANGELOG.md` 자동 생성 (git-cliff) |

---

## 최근 변경사항

<!-- CHANGELOG_START -->
## 미출시 변경사항

### CI/CD

- CI에서 의미없는 ruff --fix 제거 ([c5fe2da](https://github.com/JungYeoni/da-template/commit/c5fe2daf4c3730ddb07576937648382781dc182d))
- Ruff format --check 추가로 포맷 불일치 CI 차단 ([3e8f382](https://github.com/JungYeoni/da-template/commit/3e8f382b698896126e58e3beb2e80c9bbdf29aa3))

### style

- Black 포맷 적용 ([803edf1](https://github.com/JungYeoni/da-template/commit/803edf13cdd66469c74c6be0b664889d28664316))

### 기타

- 이슈 템플릿 name 및 제목 형식 개선 ([3fe5df7](https://github.com/JungYeoni/da-template/commit/3fe5df75c68a468c3931047f9f1c9594e570032f))
- 이슈 템플릿을 da-template 컨셉에 맞게 교체 ([a591a5d](https://github.com/JungYeoni/da-template/commit/a591a5db79004ada76ac067465dab411c6df278f))
- Uv 패키지 매니저로 전환 ([330ab72](https://github.com/JungYeoni/da-template/commit/330ab7285fb25faf8b77f8c161f5de62c67dadce))
- CHANGELOG 자동 업데이트 워크플로우 추가 (git-cliff) ([cb4777d](https://github.com/JungYeoni/da-template/commit/cb4777d787dc6481b03042084a695d5ac2eb976e))
- Docs/ gitignore 추가 ([e960045](https://github.com/JungYeoni/da-template/commit/e9600451af2c5e32d411b808fe3e72fad8fd3fee))
- Pre-commit 훅 추가 — ruff lint + format 자동 적용 ([8d33a13](https://github.com/JungYeoni/da-template/commit/8d33a13124257a6a8d290b3f5651b733fff96e02))
- Pyproject.toml에서 black 완전 제거, ruff format 설정 추가 ([2799e7b](https://github.com/JungYeoni/da-template/commit/2799e7b39b875e6fd1c30cfb03ef955e68833349))

### 문서

- CHANGELOG 자동 업데이트 [skip ci] ([fe0d00f](https://github.com/JungYeoni/da-template/commit/fe0d00f3a3db020f99f6040a6b37caee5a211284))
- CHANGELOG 자동 업데이트 [skip ci] ([53cbe0d](https://github.com/JungYeoni/da-template/commit/53cbe0d354bb6216e66753021854add6a66a3f70))
- CHANGELOG 자동 업데이트 [skip ci] ([9fcbee6](https://github.com/JungYeoni/da-template/commit/9fcbee6d6e029bfcf9e728c00d30acf89f357444))
- CHANGELOG 자동 업데이트 [skip ci] ([581b0d5](https://github.com/JungYeoni/da-template/commit/581b0d54163c8cb624eb8b783781342d32991591))
- CHANGELOG 자동 업데이트 [skip ci] ([8fd6c82](https://github.com/JungYeoni/da-template/commit/8fd6c829c4a81740172cb2ed90454b21379e2b58))
- [Docs] README에 Use this template 배지 추가 ([bd8c252](https://github.com/JungYeoni/da-template/commit/bd8c252de53f0507604338650ab59b48ad5533a8))

### 버그 수정

- Pr 제목 접두사 검사 워크플로우 삭제 ([e52463d](https://github.com/JungYeoni/da-template/commit/e52463dd9bc6f3552470cb9a6df9cc0ced29d52c))
- Cliff.toml env.GITHUB_REPO 변수 오류 수정 ([a68e29c](https://github.com/JungYeoni/da-template/commit/a68e29c9a790684ee0524e106a8106ca99d63a8c))
- Git-cliff-action Docker Buster EOL 오류 수정 ([5bb2c8b](https://github.com/JungYeoni/da-template/commit/5bb2c8bb2a06560c1cd59ce137e5b10f37b63dc9))
- Ruff I001 import 정렬 수정 ([df86e5e](https://github.com/JungYeoni/da-template/commit/df86e5ed990c20127f7ce64858e4fdc151d7ed63))
- CI에서 black 제거 — ruff로 스타일 체크 통합 ([ce5fb50](https://github.com/JungYeoni/da-template/commit/ce5fb503ad9a90c997020b12c32c51fd10a7d0dd))
- Ruff --fix로 import 정렬 자동 적용 후 체크 ([19ec536](https://github.com/JungYeoni/da-template/commit/19ec536cb161143a3cf75dd0e6ef218442456ba9))
- Ruff lint 오류 수정 ([5eedae6](https://github.com/JungYeoni/da-template/commit/5eedae647534e4c2aca44b9cd40ad52b327d33f1))
- CI 실패 수정 — build-backend 오타, GIS 의존성 분리 ([76771b1](https://github.com/JungYeoni/da-template/commit/76771b1ad21d2cbf325d849538bf4a977b7bcfd2))

### 새 기능

- README에 최근 변경사항 자동 주입 추가 ([2378fcc](https://github.com/JungYeoni/da-template/commit/2378fcc7dd765ac19519816ccd713279bbe73d5d))
- 데이터분석·ML 프로젝트 템플릿 전체 구성 ([07fbb5b](https://github.com/JungYeoni/da-template/commit/07fbb5bafcd6d59039816e44756597bfe0511820))
- 분석 슬래시 커맨드 6종 추가 및 README 작성 ([cf20bf9](https://github.com/JungYeoni/da-template/commit/cf20bf9e414eedb7f61008b3ead25b347a427b2c))
- 데이터 분석 Claude 전역 설정 초기 구성 ([b99887c](https://github.com/JungYeoni/da-template/commit/b99887ced270db03295a8dadd224732a6947a2eb))
<!-- CHANGELOG_END -->

---

## 라이선스

MIT
