# 프로젝트

정형 데이터 분석, 시계열, 회귀·인과추론, 머신러닝, GIS 분석용 Python 템플릿이다.

# 환경과 검증

- Python 3.11 이상과 `uv`를 사용한다.
- 의존성의 기준은 `pyproject.toml`과 `uv.lock`이다.
- 환경 준비: `uv sync --extra dev`
- 린트: `uv run ruff check src/ tests/`
- 포맷 검사: `uv run ruff format --check src/ tests/`
- 테스트: `uv run pytest tests/ -v`
- 변경 후 관련 검사를 실행하고 결과를 보고한다.

# 의존성

- 코드나 예시를 실행하기 전에 필요한 패키지가 `pyproject.toml`에 선언되어 있는지 확인한다.
- `polars`, `statsmodels`, `xgboost`, `lightgbm`, `plotly`, `streamlit`, `pyyaml`, `python-dotenv`는 기본 의존성이 아니다.
- 선택 패키지가 실제 작업에 필요하면 먼저 `uv add <package>`로 추가하고 `uv.lock`을 함께 갱신한다.

# 분석 원칙

- 데이터 없이 가정으로 분석하지 않는다.
- 불확실한 결론은 가능성으로 명확히 표현한다.
- 데이터 분할 기준과 난수 시드를 기록한다.
- 인코더와 스케일러는 학습 데이터에서만 학습한다.
- 시계열 lag와 rolling 피처는 미래 값을 참조하지 않는다.
- 복잡한 모델보다 해석 가능하고 재현 가능한 기준 모델을 먼저 사용한다.
- 기존 `src/` 구현과 표준 라이브러리를 확인한 뒤 새 코드를 작성한다.

# 저장 위치

- 피처 생성: `src/features/`
- 모델 학습: `src/modeling/`
- 평가: `src/evaluation/`
- 시각화: `src/visualization/`
- 계획 문서: `docs/plans/`
- 분석 보고서: `reports/`

# Git과 출력

- 사용자가 명시적으로 요청한 경우에만 커밋, 푸시, PR 생성을 진행한다.
- 커밋 메시지에 AI 공동 작성자(`Co-Authored-By`)를 추가하지 않는다.
- GitHub 이슈 템플릿을 제외하고 로그 메시지와 `print` 출력에 이모지를 사용하지 않는다.
