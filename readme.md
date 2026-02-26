# 🌪️ forecaSTAT 팀 예측 파이프라인 실행 가이드

## 📋 개요
본 파이프라인은 3가지 상황에 따라 유연하게 실행할 수 있도록 설계되었습니다. 데이터의 전처리 단계에 따라 적절한 실행 방법을 선택하세요.

---

## 🚀 실행 방법별 가이드

### 🔥 상황 1: 원본 데이터부터 전체 실행
> **가장 완전한 파이프라인** - 원본 LDAPS 데이터부터 시작하여 모든 단계를 거쳐 최종 예측까지

#### 📁 필요한 데이터 구조
```
your_project_folder/
├── ldaps/                    # 원본 LDAPS 날씨 데이터
│   ├── 20200101/
│   │   ├── 2020-01-01_00_00_00.parquet
│   │   ├── 2020-01-01_01_00_00.parquet
│   │   └── ...
│   ├── 20200102/
│   └── ...
└── target/                   # 풍력 발전량 타겟 데이터
    ├── gyeongju_target.parquet
    ├── yangyang_target.parquet
    └── yeongduk_target.parquet
```

#### 💻 실행 코드
```python
from wind_power_pipeline import WindPowerPipeline

# 📌 경로 설정 (이 부분만 수정하세요!)
BASE_PATH = "/your/project/folder"

# 파이프라인 생성 및 실행
pipeline = WindPowerPipeline(base_path=BASE_PATH, max_workers=8)

# 전체 파이프라인 실행 (모든 단계 포함)
final_result = pipeline.run_full_pipeline(
    skip_weather_processing=False,  # 원본부터 전처리
    use_processed_data=False        # 중간 데이터 사용 안함
)

print("✅ 전체 파이프라인 완료!")
print(f"📊 결과: {len(final_result)}개 예측 데이터 생성")
```

---

### ⭐ 상황 2: MAX 데이터부터 실행
> 시계열로 집계된 MAX 데이터부터 시작하여 파생변수 생성 및 모델링
> 
>  미리 넣어둔 폴도는 경주, 양양, 영덕_test_이고 쓰실땐 '_'를 꼭 빼주세요.
> 
#### 📁 필요한 데이터 구조
```
your_project_folder/
├── results/                  # MAX 시계열 데이터
│   ├── 경주_test/
│   │   └── 경주_timeseries_MAX.parquet
│   ├── 양양_test/
│   │   └── 양양_timeseries_MAX.parquet
│   └── 영덕_test/
│       └── 영덕_timeseries_MAX.parquet
└── target/                   # 풍력 발전량 타겟 데이터
    ├── gyeongju_target.parquet
    ├── yangyang_target.parquet
    └── yeongduk_target.parquet
```

### ⚡ 상황 3: 최종 처리된 데이터부터 실행 

#### 💻 실행 코드
```python
from wind_power_pipeline import WindPowerPipeline

# 📌 경로 설정 (이 부분만 수정하세요!)
BASE_PATH = "/your/project/folder"

# 파이프라인 생성 및 실행
pipeline = WindPowerPipeline(base_path=BASE_PATH, max_workers=8)

# MAX 데이터부터 실행
final_result = pipeline.run_full_pipeline(
    skip_weather_processing=True,   # MAX 데이터 사용
    use_processed_data=False        # 파생변수는 새로 생성
)

print("✅ 파이프라인 완료!")
print(f"📊 결과: {len(final_result)}개 예측 데이터 생성")
```

#### 📁 필요한 데이터 구조
```
your_project_folder/
└── 파생변수가 추가된 데이터/    # 최종 처리된 데이터
    ├── gy.parquet              # 경주 (파생변수 포함)
    ├── yy.parquet              # 양양 (파생변수 포함)
    └── yd.parquet              # 영덕 (파생변수 포함)
```

#### 💻 실행 코드
```python
from wind_power_pipeline import WindPowerPipeline

# 📌 경로 설정
BASE_PATH = "/your/project/folder"

# 파이프라인 생성 및 실행
pipeline = WindPowerPipeline(base_path=BASE_PATH, max_workers=8)

# 최종 데이터로 바로 모델링
final_result = pipeline.run_full_pipeline(
    use_processed_data=True  # 최종 데이터 직접 사용
)

print("✅ 모델링 완료!")
print(f"📊 결과: {len(final_result)}개 예측 데이터 생성")
```


## 📤 최종 결과 파일

성공적으로 실행되면 다음 파일이 생성됩니다:

```
results/
└── result.csv                # 📈 최종 예측 결과
    ├── time                  # 시간
    ├── energy_kwh           # 예측 발전량 (kWh)
    └── plant_name           # 발전소명 (경주풍력/영덕풍력/양양풍력)
```
