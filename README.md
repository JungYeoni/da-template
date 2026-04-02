# da-template

Claude Code를 데이터 분석 전문가로 설정하는 프로젝트 템플릿입니다.  
이 레포를 클론하거나 `.claude/` 폴더를 복사하면 Claude가 EDA, 시계열, GIS, ML 등 분석 워크플로우를 표준화된 방식으로 지원합니다.

## 구성

```
.claude/
├── CLAUDE.md              # 역할 및 기술 스택 기본 설정
├── rules/                 # 분석 규칙 (로드 순서 준수)
│   ├── 01_data_safety.md  # 개인정보 처리, 원본 보존
│   ├── 02_code_style.md   # PEP 8, pandas 작성 규칙
│   ├── 03_analysis_workflow.md  # EDA → 전처리 → 모델링 순서
│   ├── 04_output_format.md      # 결과 요약 형식, 숫자 표기
│   └── 05_communication.md      # 응답 언어, 불확실성 표현
└── commands/              # 슬래시 커맨드 (분석 스킬)
    ├── timeseries.md      # /timeseries
    ├── tabular.md         # /tabular
    ├── gis.md             # /gis
    ├── regression.md      # /regression
    ├── ml.md              # /ml
    └── visualization.md   # /visualization
```

## 슬래시 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/timeseries` | 시계열 분석 — ADF 정상성 검정, 계절성 분해, ARIMA, Prophet |
| `/tabular` | 테이블 EDA — 기술통계, 결측치 히트맵, 이상치 탐지, 상관관계 |
| `/gis` | 지리공간 분석 — geopandas, folium 지도, 공간 조인, DBSCAN 클러스터링 |
| `/regression` | 회귀 분석 — OLS, VIF, 잔차 진단, Ridge/Lasso 비교 |
| `/ml` | 머신러닝 파이프라인 — 전처리, 모델 비교, 하이퍼파라미터 튜닝, SHAP |
| `/visualization` | 시각화 — matplotlib/seaborn/plotly, 색약 친화 팔레트, 한국어 폰트 |

## 적용 방법

### 이 프로젝트에만 적용

```bash
git clone https://github.com/LeeJungYeon/da-template.git
cd your-project
cp -r da-template/.claude .
```

### 모든 프로젝트에 전역 적용

```bash
cp -r da-template/.claude/rules ~/.claude/rules
cp -r da-template/.claude/commands ~/.claude/commands
cp da-template/.claude/CLAUDE.md ~/.claude/CLAUDE.md
```

> 전역 적용 시 기존 `~/.claude/CLAUDE.md`가 있다면 내용을 병합하세요.

## 주요 규칙 요약

**데이터 안전성**
- PII 컬럼은 분석 전 마스킹/제거 먼저 제안
- 원본 데이터프레임 직접 수정 금지 → `df_clean = df.copy()` 패턴
- 대용량 파일(>1GB)은 청크 처리 방식 기본 제안

**분석 워크플로우**
- 데이터 로드 → EDA → 전처리 → 분석/모델링 → 결과 해석 순서 준수
- EDA 없이 바로 모델링 요청 시 EDA 먼저 제안
- `random_state=42` 포함으로 재현성 확보

**코드 스타일**
- PEP 8, 라인 길이 최대 100자
- pandas 메서드 체이닝은 괄호로 감싸서 줄바꿈
- `iterrows()` 지양, 벡터 연산 우선

## 기술 스택

- **언어**: Python 3.10+
- **데이터**: pandas, polars
- **시각화**: matplotlib, seaborn, plotly
- **통계**: scipy, statsmodels
- **ML**: scikit-learn, xgboost, lightgbm
- **GIS**: geopandas, folium, shapely
- **환경**: Jupyter Notebook / VS Code

## 라이선스

MIT
