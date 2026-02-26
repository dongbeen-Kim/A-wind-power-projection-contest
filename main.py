from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import os
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional
from dataclasses import dataclass

import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd

@dataclass
class LocationConfig:
    name: str
    lat: float
    lon: float
    lat_range: float
    lon_range: float
    start_date: str
    end_date: str


class WindPowerPipeline:
    def __init__(self, base_path: str, max_workers: int = 8):
        """
        풍력 발전 예측 파이프라인 초기화

        Args:
            base_path (str): 기본 작업 디렉토리 경로
            max_workers (int): 멀티프로세싱 워커 수
        """
        self.base_path = Path(base_path)
        self.max_workers = max_workers
        self.schema_map = {}

        # 하위 디렉토리 설정
        self.ldaps_path = self.base_path / "ldaps"
        self.output_path = self.base_path / "output"
        self.target_path = self.base_path / "target"
        self.result_path = self.base_path / "results"
        self.derived_data_path = self.base_path / "파생변수추가된데이터"

        # 결과 디렉토리 생성
        self.result_path.mkdir(exist_ok=True)

    def _get_location_configs(self) -> Dict[str, LocationConfig]:
        """지역별 설정 정보 반환"""
        return {
            'gyeongju': LocationConfig(
                name='경주',
                lat=35.7149,
                lon=129.3693,
                lat_range=0.015,
                lon_range=0.015,
                start_date='2020-01-01',
                end_date='2024-12-31'
            ),
            'yangyang': LocationConfig(
                name='양양',
                lat=37.9330943,
                lon=128.6943946,
                lat_range=0.02,
                lon_range=0.02,
                start_date='2024-04-01',
                end_date='2025-03-31'
            ),
            'yeongdeok': LocationConfig(
                name='영덕',
                lat=36.4198685,
                lon=129.3960048,
                lat_range=0.015,
                lon_range=0.015,
                start_date='2024-04-01',
                end_date='2025-03-31'
            )
        }

    def _date_in_range(self, date_str: str, start: str, end: str) -> bool:
        """날짜가 범위 내에 있는지 확인"""
        date = dt.datetime.strptime(date_str, "%Y%m%d").date()
        return dt.date.fromisoformat(start) <= date <= dt.date.fromisoformat(end)

    def _filter_lat_lon_lazy(self, df: pl.LazyFrame, config: LocationConfig) -> pl.LazyFrame:
        """위도/경도 필터링"""
        lat_min = config.lat - config.lat_range
        lat_max = config.lat + config.lat_range
        lon_min = config.lon - config.lon_range
        lon_max = config.lon + config.lon_range

        return df.filter(
            (pl.col("latitude") >= lat_min) & (pl.col("latitude") <= lat_max) &
            (pl.col("longitude") >= lon_min) & (pl.col("longitude") <= lon_max)
        )

    def _get_target_dates(self, config: LocationConfig) -> List[str]:
        """처리할 날짜 목록 반환"""
        root_folder = str(self.ldaps_path)
        output_folder = str(self.output_path)

        all_dates = [d for d in os.listdir(root_folder)
                     if os.path.isdir(os.path.join(root_folder, d)) and d.isdigit()]
        done_dates = [d for d in os.listdir(output_folder)
                      if os.path.isdir(os.path.join(output_folder, d)) and d.isdigit()] if os.path.exists(
            output_folder) else []
        target_dates = sorted([d for d in all_dates if d not in done_dates])

        return [d for d in target_dates
                if self._date_in_range(d, config.start_date, config.end_date)]

    def process_weather_data(self):
        """날씨 데이터 전처리"""
        print("=== 날씨 데이터 전처리 시작 ===")

        processor = WeatherDataProcessor(max_workers=self.max_workers)
        locations = ['gyeongju', 'yangyang', 'yeongdeok']

        for location in locations:
            try:
                config = processor._get_location_configs()[location]
                location_result_path = self.result_path / f"{config.name}_test"
                location_result_path.mkdir(exist_ok=True)

                parquet_path = location_result_path / f"{config.name}_timeseries_MAX.parquet"
                csv_path = location_result_path / f"{config.name}_timeseries_MAX.csv"

                processor.process_location_data(
                    location_key=location,
                    root_folder=str(self.ldaps_path),
                    output_folder=str(self.output_path),
                    output_parquet=str(parquet_path),
                    output_csv=str(csv_path)
                )

            except Exception as e:
                print(f"{location} 처리 중 오류 발생: {e}")

    def load_weather_data(self):
        """원본 날씨 데이터 로드 (MAX 데이터)"""
        print("=== 날씨 데이터 로드 ===")

        # 먼저 새로 생성된 MAX 데이터가 있는지 확인
        max_files = {
            'gyeongju': self.result_path / "경주_test/경주_timeseries_MAX.parquet",
            'yangyang': self.result_path / "양양_test/양양_timeseries_MAX.parquet",
            'yeongdeok': self.result_path / "영덕_test/영덕_timeseries_MAX.parquet"
        }

        # MAX 파일들이 모두 존재하는지 확인
        all_max_exist = all(path.exists() for path in max_files.values())

        if all_max_exist:
            print("생성된 MAX 데이터 사용")
            df1 = pd.read_parquet(max_files['gyeongju'])
            df2 = pd.read_parquet(max_files['yangyang'])
            df3 = pd.read_parquet(max_files['yeongdeok'])
            print("MAX 데이터 로드 완료")
        else:
            print("MAX 데이터가 없습니다. 날씨 데이터 전처리를 먼저 실행해주세요.")
            raise FileNotFoundError("MAX 시계열 데이터가 존재하지 않습니다.")

        return df1, df2, df3

    def load_target_data(self):
        """타겟 데이터 로드 및 전처리"""
        print("=== 타겟 데이터 처리 ===")

        dataframes = []
        target_files = list(self.target_path.glob('*.parquet'))

        if not target_files:
            raise FileNotFoundError(f"타겟 데이터를 찾을 수 없습니다: {self.target_path}")

        for file in target_files:
            df = pd.read_parquet(file)
            filename_suffix = file.stem.split('_')[-1]
            df['filename'] = filename_suffix
            dataframes.append(df)

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
        else:
            merged_df = pd.DataFrame()

        merged_df.drop(['구분', '시간', 'energy_mwh', 'plant_name', 'period_hours'],
                       axis=1, inplace=True, errors='ignore')
        merged_df['date'] = merged_df['end_datetime'].dt.date
        merged_df['hour'] = merged_df['end_datetime'].dt.hour

        # 타겟 데이터 저장
        target_output_path = self.result_path / 'target.parquet'
        merged_df.to_parquet(target_output_path, engine='pyarrow', index=False)

        return merged_df

    def create_features(self, df1, df2, df3, target_data):
        """파생변수 생성 및 데이터 결합"""
        print("=== 파생변수 생성 및 데이터 결합 ===")

        # 타겟 데이터와 날씨 데이터 결합
        target_data['end_datetime'] = target_data['end_datetime'].dt.tz_localize(None)
        df1['time'] = pd.to_datetime(df1['time'])
        df2['time'] = pd.to_datetime(df2['time'])
        df3['time'] = pd.to_datetime(df3['time'])

        # 각 지역별 데이터 결합
        gy = pd.merge(df1, target_data[target_data['filename'] == 'gyeongju'],
                      left_on='time', right_on='end_datetime', how='left')
        gy.drop(['end_datetime', 'filename', 'date', 'hour'], axis=1, inplace=True, errors='ignore')

        yd = pd.merge(df3, target_data[target_data['filename'] == 'yeongduk'],
                      left_on='time', right_on='end_datetime', how='left')
        yd.drop(['end_datetime', 'filename', 'date', 'hour'], axis=1, inplace=True, errors='ignore')

        yy = pd.merge(df2, target_data[target_data['filename'] == 'yangyang'],
                      left_on='time', right_on='end_datetime', how='left')
        yy.drop(['end_datetime', 'filename', 'date', 'hour'], axis=1, inplace=True, errors='ignore')

        # 파생변수 생성
        def calculate_absolute_humidity_vec(temp_K, humidity):
            temp_C = temp_K - 273.15
            e_s = 6.112 * np.exp((17.67 * temp_C) / (temp_C + 243.5))
            e = humidity * e_s / 100
            AH = (216.7 * e) / temp_K
            return AH

        def add_derived_features(df, turbine_rad):
            # 절대습도
            df['absolute_humidity'] = calculate_absolute_humidity_vec(df['ta_1p5m'], df['rh_1p5m'])

            # 터빈 면적
            df['turbine_area'] = np.pi * turbine_rad ** 2

            # 돌풍속도
            df['storm_speed'] = np.sqrt(df['fvmax_50m'] ** 2 + df['fvmin_50m'] ** 2)

            # 공기 밀도
            R = 287.05
            df['air_density'] = (df['pmsl'] * 100) / (R * df['ta_1p5m'])

            # 풍속
            df['wind_speed'] = np.sqrt(df['uws_10m'] ** 2 + df['vws_10m'] ** 2)
            df['wind_speed_squared'] = df['wind_speed'] ** 2

            # 풍향
            df['wind_direction'] = (np.degrees(np.arctan2(df['vws_10m'], df['uws_10m'])) + 360) % 360
            df['storm_direction'] = (np.degrees(np.arctan2(df['fvmin_50m'], df['fvmax_50m'])) + 360) % 360
            df['wind_direction_diff'] = abs(df['wind_direction'] - df['storm_direction'])

            # 풍속-풍향 상호작용
            df['wind_speed_direction_interaction'] = df['wind_speed'] * np.cos(np.radians(df['wind_direction']))

            # 난류 강도
            df['turbulence_intensity'] = (df['fvmax_50m'] - df['fvmin_50m']) / (df['wind_speed'] + 0.1)

            # 풍력 에너지 계산
            df['wind_energy'] = 0.5 * df['air_density'] * df['turbine_area'] * df['wind_speed'] ** 3 / 1000 / 3600

            # 날짜 관련 파생변수
            df['year'] = df['time'].dt.year
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day
            df['hour'] = df['time'].dt.hour

            return df

        # 각 지역별 파생변수 생성
        turbine_rad_gy = 56.5
        turbine_rad_yd = 75.5
        turbine_rad_yy = 67.85

        gy = add_derived_features(gy, turbine_rad_gy)
        yd = add_derived_features(yd, turbine_rad_yd)
        yy = add_derived_features(yy, turbine_rad_yy)

        return gy, yd, yy

    def preprocess_data(self, gy, yd, yy):
        """데이터 전처리"""
        print("=== 데이터 전처리 ===")

        # 경주 데이터 전처리
        gy = gy[gy['time'] != '2020-01-01 00:00:00']
        gy = gy[~gy.drop(['energy_kwh'], axis=1).isna().any(axis=1)]

        # 영덕 데이터 전처리 (보간)
        yd = yd.set_index('time', drop=False)
        full_date_range = pd.date_range(start="2024-04-01 00:00:00", end="2025-03-31 23:00:00", freq='H')
        yd = yd.reindex(full_date_range)
        columns_to_interpolate = [col for col in yd.columns if col not in ['energy_kwh', 'time']]
        yd[columns_to_interpolate] = yd[columns_to_interpolate].interpolate(method='linear')

        # 양양 데이터 전처리 (보간)
        yy = yy[yy['time'] != '2024-04-01 00:00:00']
        yy = yy.set_index('time', drop=False)
        full_date_range = pd.date_range(start="2024-04-01 01:00:00", end="2025-03-31 23:00:00", freq='H')
        yy = yy.reindex(full_date_range)
        columns_to_interpolate = [col for col in yy.columns if col not in ['energy_kwh', 'time']]
        yy[columns_to_interpolate] = yy[columns_to_interpolate].interpolate(method='linear')

        # 전처리된 데이터 저장
        gy.to_parquet(self.result_path / 'mid_output_gy.parquet')
        yy.to_parquet(self.result_path / 'mid_output_yy.parquet')
        yd.to_parquet(self.result_path / 'mid_output_yd.parquet')

        print("전처리 완료")
        return gy, yd, yy

    def train_models_and_predict(self, gy, yd, yy):
        """모델 학습 및 예측 (가중치 손실 + 정산금 최적화 + 시간대 분리 모델)"""
        print("=== 모델 학습 및 예측 (개선 포인트 1,2,3 적용) ===")

        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint, uniform
        import numpy as np
        from functools import partial

        # 발전소 용량을 데이터에서 자동 추정
        CAPACITY = {
            '경주풍력': gy['energy_kwh'].max() * 1.04,
            '영덕풍력': yd['energy_kwh'].max() * 1.05,
            '양양풍력': yy['energy_kwh'].max() * 1.05
        }

        print(f"추정 용량: {CAPACITY}")

        def custom_weighted_loss(y_true, y_pred, capacity):
            """중요 시간대 가중치 적용 손실 함수"""
            errors = np.abs(y_true - y_pred)
            weights = np.where(y_true >= 0.1 * capacity, 2.0, 1.0)
            return np.mean(weights * errors)

        def bias_correction(y_pred, y_train, capacity):
            """정산금 최적화를 위한 바이어스 보정"""
            mean_pred = np.mean(y_pred)
            mean_actual = np.mean(y_train)
            bias = mean_actual - mean_pred

            # y_corrected = y_pred + bias * 0.5
            y_corrected = y_pred
            high_generation_mask = y_corrected >= 0.1 * capacity
            y_corrected[high_generation_mask] *= 1.05
            y_corrected = np.clip(y_corrected, 0, capacity)

            return y_corrected
        # ✅ 5️⃣ 추가: 예측 smoothing (후처리)
        def smooth_predictions(y_pred, window=3):
            """
            급격한 변화를 완화하는 예측 smoothing 함수
            이동평균 필터로 예측 변동을 완화하여 MAE 안정화
            """
            if len(y_pred) < window:
                return y_pred
            return np.convolve(y_pred, np.ones(window)/window, mode='same')

        def custom_scorer(estimator, X, y_true, capacity):
            """커스텀 스코어링 함수"""
            y_pred = estimator.predict(X)
            return -custom_weighted_loss(y_true, y_pred, capacity)

        def train_dual_model(X_train, y_train, X_test, capacity, model_type='rf'):
            """
            중요 시간대와 일반 시간대를 분리하여 학습하는 이중 모델
            """
            threshold = 0.1 * capacity

            # numpy array로 변환 (indexing을 위해)
            y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train

            # 중요 시간대 (발전량 >= 10%)
            high_mask_train = y_train_arr >= threshold
            X_train_high = X_train[high_mask_train]
            y_train_high = y_train_arr[high_mask_train]

            # 일반 시간대 (발전량 < 10%)
            low_mask_train = y_train_arr < threshold
            X_train_low = X_train[low_mask_train]
            y_train_low = y_train_arr[low_mask_train]

            print(f"  중요 시간대 샘플: {len(y_train_high)} / 일반 시간대 샘플: {len(y_train_low)}")

            # 모델 파라미터 설정
            if model_type == 'rf':
                high_params = {
                    'n_estimators': randint(300, 700),
                    'max_depth': [20, 25, 30, None],
                    'min_samples_split': randint(2, 10),
                    'min_samples_leaf': randint(1, 5),
                    'max_features': ['sqrt', 'log2']
                }
                low_params = {
                    'n_estimators': randint(100, 400),
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': randint(2, 15),
                    'min_samples_leaf': randint(1, 8),
                    'max_features': ['sqrt', 'log2']
                }
                base_model_high = RandomForestRegressor(random_state=42, n_jobs=-1)
                base_model_low = RandomForestRegressor(random_state=42, n_jobs=-1)
            elif model_type == 'lgbm':
                high_params = {
                    'n_estimators': randint(300, 700),
                    'max_depth': randint(8, 20),
                    'learning_rate': uniform(0.01, 0.15),
                    'num_leaves': randint(50, 150),
                    'min_child_samples': randint(5, 50),
                    'subsample': uniform(0.75, 0.25),
                    'colsample_bytree': uniform(0.75, 0.25)
                }
                low_params = {
                    'n_estimators': randint(100, 400),
                    'max_depth': randint(5, 15),
                    'learning_rate': uniform(0.01, 0.2),
                    'num_leaves': randint(20, 100),
                    'min_child_samples': randint(10, 80),
                    'subsample': uniform(0.7, 0.3),
                    'colsample_bytree': uniform(0.7, 0.3)
                }
                base_model_high = LGBMRegressor(random_state=42, verbose=-1)
                base_model_low = LGBMRegressor(random_state=42, verbose=-1)
            # 💡 XGBoost 추가
            elif model_type == 'xgb':
                high_params = {
                    'n_estimators': randint(300, 700),
                    'max_depth': randint(8, 20),
                    'learning_rate': uniform(0.01, 0.15),
                    'subsample': uniform(0.75, 0.25),
                    'colsample_bytree': uniform(0.75, 0.25)
                }
                low_params = {
                    'n_estimators': randint(100, 400),
                    'max_depth': randint(5, 15),
                    'learning_rate': uniform(0.01, 0.2),
                    'subsample': uniform(0.7, 0.3),
                    'colsample_bytree': uniform(0.7, 0.3)
                }
                base_model_high = xgb.XGBRegressor(random_state=42, n_jobs=-1)
                base_model_low = xgb.XGBRegressor(random_state=42, n_jobs=-1)

            # 💡 SVR 추가 (파라미터 공간은 좁게 설정하거나, Grid Search 고려)
            elif model_type == 'svr':
                high_params = {
                    'C': uniform(0.5, 5.0), # 규제 파라미터
                    'gamma': ['scale', 'auto', uniform(0.001, 0.1)], # 커널 계수
                    'epsilon': uniform(0.01, 0.5) # 허용 오차
                }
                low_params = {
                    'C': uniform(0.1, 2.0),
                    'gamma': ['scale', 'auto', uniform(0.0001, 0.01)],
                    'epsilon': uniform(0.05, 0.2)
                }
                # SVR은 대규모 데이터에서 학습 시간이 매우 오래 걸릴 수 있습니다.
                base_model_high = SVR(kernel='rbf')
                base_model_low = SVR(kernel='rbf')

            # 💡 Ridge 추가 (간단하고 빠른 선형 모델)
            elif model_type == 'ridge':
                high_params = {
                    'alpha': uniform(0.1, 10.0), # 규제 강도
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
                }
                low_params = {
                    'alpha': uniform(0.01, 5.0),
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
                }
                # Ridge는 n_jobs를 지원하지 않으므로 주의
                base_model_high = Ridge(random_state=42)
                base_model_low = Ridge(random_state=42)

            # 💡 MLPRegressor 추가 (인공신경망)
            elif model_type == 'mlp':
                high_params = {
                    'hidden_layer_sizes': [(randint(50, 200).rvs(),), (randint(50, 150).rvs(), randint(20, 80).rvs())],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam'],
                    'learning_rate_init': uniform(0.0001, 0.01),
                    'max_iter': randint(300, 800)
                }
                low_params = {
                    'hidden_layer_sizes': [(randint(20, 100).rvs(),)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam'],
                    'learning_rate_init': uniform(0.001, 0.05),
                    'max_iter': randint(100, 500)
                }
                base_model_high = MLPRegressor(random_state=42)
                base_model_low = MLPRegressor(random_state=42)

            # 💡 ElasticNet 추가 (L1, L2 규제가 혼합된 선형 모델)
            elif model_type == 'elasticnet':
                high_params = {
                    'alpha': uniform(0.001, 50.0), # 전체 규제 강도
                    'l1_ratio': uniform(0.0, 1.0), # L1 규제의 혼합 비율 (0: L2, 1: L1)
                    'selection': ['cyclic', 'random']
                }
                low_params = {
                    'alpha': uniform(0.001, 20.0),
                    'l1_ratio': uniform(0.0, 1.0),
                    'selection': ['cyclic', 'random']
                }
                # ElasticNet은 n_jobs를 지원하지 않으므로 주의
                base_model_high = ElasticNet(random_state=42)
                base_model_low = ElasticNet(random_state=42)

            # 💡 Lasso 추가 (L1 규제를 사용하는 선형 모델 - 특성 선택 효과)
            elif model_type == 'lasso':
                high_params = {
                    'alpha': uniform(0.1, 10.0), # 규제 강도
                    'selection': ['cyclic', 'random']
                }
                low_params = {
                    'alpha': uniform(0.01, 5.0),
                    'selection': ['cyclic', 'random']
                }
                # Lasso는 n_jobs를 지원하지 않으므로 주의
                base_model_high = Lasso(random_state=42)
                base_model_low = Lasso(random_state=42)

            else:
                raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

            # 중요 시간대 모델 학습
            print("  중요 시간대 모델 학습 중...")
            search_high = RandomizedSearchCV(
                base_model_high,
                high_params,
                n_iter=20,
                cv=3,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            search_high.fit(X_train_high, y_train_high)
            print(f"  중요 시간대 최적 점수: {-search_high.best_score_:.2f}")

            # 일반 시간대 모델 학습
            print("  일반 시간대 모델 학습 중...")
            search_low = RandomizedSearchCV(
            base_model_low,
            low_params,
            n_iter=15 if model_type not in ['svr', 'mlp', 'ridge'] else 7, # SVR, MLP는 튜닝 횟수 줄임
            cv=3,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1 if model_type not in ['svr', 'ridge', 'mlp'] else 1, # n_jobs 지원하지 않는 모델 처리
            verbose=0
            )
            search_low.fit(X_train_low, y_train_low)
            print(f"  일반 시간대 최적 점수: {-search_low.best_score_:.2f}")

            # 테스트 데이터 예측
            y_pred_high = search_high.best_estimator_.predict(X_test)
            y_pred_low = search_low.best_estimator_.predict(X_test)

            # 중요 시간대 예측값은 높은 모델, 낮은 예측값은 낮은 모델 사용
            y_pred_final = np.where(y_pred_high >= threshold, y_pred_high, y_pred_low)

            return y_pred_final, search_high.best_estimator_, search_low.best_estimator_

        results_list = []

        # ========== 1. 경주 모델 (분리 모델 적용) ==========
        print("\n경주 모델 학습 중 (분리 모델)...")
        train_data = gy[gy['year'] <= 2023]
        test_data = gy[gy['year'] == 2024]

        features = [col for col in gy.columns if col not in ['time', 'energy_kwh']]

        X_train = train_data[features]  # DataFrame 유지
        y_train = train_data['energy_kwh']  # Series 유지
        X_test = test_data[features]  # DataFrame 유지

        capacity_gy = CAPACITY['경주풍력']

        # 분리 모델로 학습 및 예측
        predict_energy_kwh, model_high, model_low = train_dual_model(X_train, y_train, X_test, capacity_gy, model_type='rf')
        # predict_energy_kwh, model_high, model_low = train_dual_model(X_train, y_train, X_test, capacity_gy, model_type='lgbm')

        # 바이어스 보정
        predict_energy_kwh = bias_correction(predict_energy_kwh, y_train.values, capacity_gy)

        # ✅ 예측 후 smoothing 적용
        predict_energy_kwh = smooth_predictions(predict_energy_kwh, window=5)
        result_gy = pd.DataFrame({
            'time': test_data['time'].values,
            'energy_kwh': predict_energy_kwh,
            'plant_name': '경주풍력'
        })
        results_list.append(result_gy)
        print("경주 모델 완료")

        # ========== 2. 영덕 모델 (분리 모델 적용) ==========
        print("\n영덕 모델 학습 중 (분리 모델)...")
        train_mask = yd['month'] % 2 == 1
        test_mask = yd['month'] % 2 == 0

        X_train = yd[train_mask].drop(['time', 'energy_kwh'], axis=1)  # DataFrame 유지
        X_test = yd[test_mask].drop(['time', 'energy_kwh'], axis=1)  # DataFrame 유지
        y_train = yd.loc[train_mask, 'energy_kwh']  # Series 유지

        capacity_yd = CAPACITY['영덕풍력']

        # 분리 모델로 학습 및 예측
        y_pred_yd, model_high, model_low = train_dual_model(X_train, y_train, X_test, capacity_yd, model_type='elasticnet')
        # y_pred_yd, model_high, model_low = train_dual_model(X_train, y_train, X_test, capacity_yd, model_type='lgbm')

        # 바이어스 보정
        y_pred_yd = bias_correction(y_pred_yd, y_train.values, capacity_yd)
        y_pred_yd = smooth_predictions(y_pred_yd, window=5)

        result_yd = pd.DataFrame({
            'time': yd.loc[test_mask, 'time'].values,
            'energy_kwh': y_pred_yd,
            'plant_name': '영덕풍력'
        })
        results_list.append(result_yd)
        print("영덕 모델 완료")

        # ========== 3. 양양 모델 (분리 모델 적용) ==========
        print("\n양양 모델 학습 중 (분리 모델)...")
        train_mask = yy['month'] % 2 == 0
        test_mask = yy['month'] % 2 == 1

        X_train = yy[train_mask].drop(['time', 'energy_kwh'], axis=1)  # DataFrame 유지
        X_test = yy[test_mask].drop(['time', 'energy_kwh'], axis=1)  # DataFrame 유지
        y_train = yy.loc[train_mask, 'energy_kwh']  # Series 유지

        capacity_yy = CAPACITY['양양풍력']

        # 분리 모델로 학습 및 예측
        # y_pred_yy, model_high, model_low = train_dual_model(X_train, y_train, X_test, capacity_yy, model_type='lgbm')
        y_pred_yy, model_high, model_low = train_dual_model(X_train, y_train, X_test, capacity_yy, model_type='elasticnet')

        # 바이어스 보정
        y_pred_yy = bias_correction(y_pred_yy, y_train.values, capacity_yy)
        y_pred_yy = smooth_predictions(y_pred_yy, window=5)
        result_yy = pd.DataFrame({
            'time': yy.loc[test_mask, 'time'].values,
            'energy_kwh': y_pred_yy,
            'plant_name': '양양풍력'
        })
        results_list.append(result_yy)
        print("양양 모델 완료")

        print("\n=== 모든 모델 학습 완료 ===")
        return results_list



    def save_results(self, results_list):
        """최종 결과 저장"""
        print("=== 결과 저장 ===")

        final_df = pd.concat(results_list, ignore_index=True)
        result_file_path = self.result_path / 'result.csv'
        final_df.to_csv(result_file_path, index=False)

        print(f"결과 저장 완료: {result_file_path}")
        print(f"총 예측 데이터 수: {len(final_df)}")
        return final_df

    def load_processed_data(self):
        """이미 파생변수가 추가된 최종 데이터 로드"""
        print("=== 최종 처리된 데이터 로드 ===")

        processed_files = {
            'gy': self.derived_data_path / 'gy.parquet',
            'yy': self.derived_data_path / 'yy.parquet',
            'yd': self.derived_data_path / 'yd.parquet'
        }

        # 모든 파일이 존재하는지 확인
        all_exist = all(path.exists() for path in processed_files.values())

        if all_exist:
            gy = pd.read_parquet(processed_files['gy'])
            yy = pd.read_parquet(processed_files['yy'])
            yd = pd.read_parquet(processed_files['yd'])
            print("최종 처리된 데이터 로드 완료")
            return gy, yy, yd
        else:
            missing = [name for name, path in processed_files.items() if not path.exists()]
            print(f"최종 처리 데이터 없음: {missing}")
            return None, None, None

    def run_full_pipeline(self, skip_weather_processing=True, use_processed_data=True):
        """전체 파이프라인 실행"""
        print("=== 풍력 발전량 예측 파이프라인 시작 ===")
        print(f"작업 디렉토리: {self.base_path}")

        # 1. 이미 최종 처리된 데이터가 있는지 확인
        if use_processed_data:
            gy, yy, yd = self.load_processed_data()
            if gy is not None and yy is not None and yd is not None:
                print("최종 처리된 데이터 사용 - 바로 모델링 단계로 진행")
                # 5. 모델 학습 및 예측
                results_list = self.train_models_and_predict(gy, yd, yy)
                # 6. 결과 저장
                final_df = self.save_results(results_list)
                print("=== 파이프라인 완료 ===")
                return final_df

        # 2. 날씨 데이터 전처리 (선택적)
        if not skip_weather_processing:
            self.process_weather_data()

        # 3. 원본 날씨 데이터 로드 (MAX 데이터)
        df1, df2, df3 = self.load_weather_data()
        target_data = self.load_target_data()

        # 4. 파생변수 생성 및 데이터 결합
        gy, yd, yy = self.create_features(df1, df2, df3, target_data)

        # 5. 데이터 전처리
        gy, yd, yy = self.preprocess_data(gy, yd, yy)

        # 6. 모델 학습 및 예측
        results_list = self.train_models_and_predict(gy, yd, yy)

        # 7. 결과 저장
        final_df = self.save_results(results_list)

        print("=== 파이프라인 완료 ===")
        return final_df


# WeatherDataProcessor 클래스 (원본 코드에서 가져옴)
class WeatherDataProcessor:
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.schema_map = {}

    def _get_location_configs(self) -> Dict[str, LocationConfig]:
        return {
            'gyeongju': LocationConfig(
                name='경주',
                lat=35.7149,
                lon=129.3693,
                lat_range=0.015,
                lon_range=0.015,
                start_date='2020-01-01',
                end_date='2024-12-31'
            ),
            'yangyang': LocationConfig(
                name='양양',
                lat=37.9330943,
                lon=128.6943946,
                lat_range=0.02,
                lon_range=0.02,
                start_date='2024-04-01',
                end_date='2025-03-31'
            ),
            'yeongdeok': LocationConfig(
                name='영덕',
                lat=36.4198685,
                lon=129.3960048,
                lat_range=0.015,
                lon_range=0.015,
                start_date='2024-04-01',
                end_date='2025-03-31'
            )
        }

    def _date_in_range(self, date_str: str, start: str, end: str) -> bool:
        date = dt.datetime.strptime(date_str, "%Y%m%d").date()
        return dt.date.fromisoformat(start) <= date <= dt.date.fromisoformat(end)

    def _filter_lat_lon_lazy(self, df: pl.LazyFrame, config: LocationConfig) -> pl.LazyFrame:
        lat_min = config.lat - config.lat_range
        lat_max = config.lat + config.lat_range
        lon_min = config.lon - config.lon_range
        lon_max = config.lon + config.lon_range

        return df.filter(
            (pl.col("latitude") >= lat_min) & (pl.col("latitude") <= lat_max) &
            (pl.col("longitude") >= lon_min) & (pl.col("longitude") <= lon_max)
        )

    def _get_target_dates(self, root_folder: str, output_folder: str, config: LocationConfig) -> List[str]:
        all_dates = [d for d in os.listdir(root_folder)
                     if os.path.isdir(os.path.join(root_folder, d)) and d.isdigit()]
        done_dates = [d for d in os.listdir(output_folder)
                      if os.path.isdir(os.path.join(output_folder, d)) and d.isdigit()] if os.path.exists(
            output_folder) else []
        target_dates = sorted([d for d in all_dates if d not in done_dates])

        return [d for d in target_dates
                if self._date_in_range(d, config.start_date, config.end_date)]

    def _collect_parquet_files(self, root_folder: str, target_dates: List[str]) -> List[str]:
        all_parquet_files = []
        for date_folder in tqdm(target_dates, desc="Scanning folders", unit="folder"):
            date_path = os.path.join(root_folder, date_folder)
            for root, dirs, files in os.walk(date_path):
                for file in files:
                    if file.endswith(".parquet"):
                        folder_date = date_folder
                        file_date = file.split('_')[0].replace('-', '')
                        if file_date == folder_date:
                            all_parquet_files.append(os.path.join(root, file))
        return all_parquet_files

    def _process_single_file(self, file_path: str, root_folder: str,
                             output_folder: str, config: LocationConfig) -> str:
        lazy_df = pl.scan_parquet(file_path)
        filtered_df = self._filter_lat_lon_lazy(lazy_df, config).collect()
        relative_path = os.path.relpath(file_path, root_folder)
        save_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        filtered_df.write_parquet(save_path)
        return file_path

    def _process_file_for_timeseries(self, file_path: str) -> Optional[pl.DataFrame]:
        try:
            file_name = os.path.basename(file_path).replace('.parquet', '')
            time_val = datetime.strptime(file_name, "%Y-%m-%d_%H_%M_%S")

            lazy_df = pl.scan_parquet(file_path)
            schema = lazy_df.collect_schema()
            schema_signature = str([(c, str(dtype)) for c, dtype in schema.items()])
            self.schema_map.setdefault(schema_signature, []).append(file_path)

            casted_df = lazy_df.with_columns([
                pl.col(c).cast(pl.Float32) for c, dtype in schema.items()
                if dtype in (pl.Float64,)
            ])
            num_cols = [c for c, dtype in casted_df.collect_schema().items()
                        if dtype in (pl.Float64, pl.Int64, pl.Float32)]
            max_df = casted_df.select([pl.col(c).max().alias(c) for c in num_cols])
            max_df = max_df.with_columns(pl.lit(time_val).alias("time"))

            return max_df

        except Exception as e:
            print(f"에러 발생: {file_path} -> {e}")
            return None

    def _filter_files_by_date_range(self, root_folder: str, start_date: str, end_date: str) -> List[str]:
        all_files = []
        date_folders = sorted(os.listdir(root_folder))
        for date_folder in tqdm(date_folders, desc="Collecting files"):
            date_path = os.path.join(root_folder, date_folder)
            if not os.path.isdir(date_path) or not date_folder.isdigit():
                continue
            if not self._date_in_range_yyyymmdd(date_folder, start_date, end_date):
                continue
            for root, _, files in os.walk(date_path):
                for f in files:
                    if f.endswith(".parquet"):
                        all_files.append(os.path.join(root, f))
        return all_files

    def _date_in_range_yyyymmdd(self, date_str: str, start: str, end: str) -> bool:
        d = datetime.strptime(date_str, "%Y%m%d").date()
        return (datetime.strptime(start, "%Y%m%d").date() <= d <=
                datetime.strptime(end, "%Y%m%d").date())

    def _print_schema_info(self):
        print("\n=== 스키마 그룹별 파일 분포 ===")
        for i, (sig, files) in enumerate(self.schema_map.items(), 1):
            print(f"\n[스키마 {i}] ({len(files)}개 파일)")
            print("샘플 스키마:", sig[:200] + "..." if len(sig) > 200 else sig)
            if len(files) <= 5:
                for f in files:
                    print("  ", f)
            else:
                print("  ... (총", len(files), "개)")

    def process_location_data(self, location_key: str, root_folder: str,
                              output_folder: str, output_parquet: str, output_csv: str):
        configs = self._get_location_configs()
        if location_key not in configs:
            raise ValueError(f"지원하지 않는 지역: {location_key}")

        config = configs[location_key]
        print(f"\n=== {config.name} 데이터 처리 시작 ===")
        print("root_folder 전체 내용:", os.listdir(root_folder))
        os.makedirs(output_folder, exist_ok=True)

        target_dates = self._get_target_dates(root_folder, output_folder, config)
        all_parquet_files = self._collect_parquet_files(root_folder, target_dates)
        print(f"총 {len(all_parquet_files)}개 파일을 처리")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(
                executor.map(
                    lambda fp: self._process_single_file(fp, root_folder, output_folder, config),
                    all_parquet_files
                ),
                total=len(all_parquet_files),
                desc="Filtering files"
            ))

        print(f"\n=== {config.name} 시계열 데이터 생성 ===")
        start_date_yyyymmdd = config.start_date.replace('-', '')
        end_date_yyyymmdd = config.end_date.replace('-', '')

        all_files = self._filter_files_by_date_range(
            output_folder, start_date_yyyymmdd, end_date_yyyymmdd
        )
        print(f"총 {len(all_files)}개 파일 처리 예정.")

        lazy_results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, os.cpu_count())) as executor:
            futures = {executor.submit(self._process_file_for_timeseries, fp): fp for fp in all_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                res = future.result()
                if res is not None:
                    lazy_results.append(res)

        self._print_schema_info()
        if lazy_results:
            final_lazy_df = pl.concat(lazy_results, how="vertical").sort("time")
            final_df = final_lazy_df.collect()

            final_df.write_parquet(output_parquet)
            final_df.write_csv(output_csv)

            print(f"\n완료: {len(final_df)}개 시간대 데이터 저장")
            print(f"- Parquet: {output_parquet}")
            print(f"- CSV: {output_csv}")
        else:
            print("처리할 데이터가 없습니다.")


def main():
    """
    메인 실행 함수
    사용자가 기본 경로만 설정하면 전체 파이프라인이 실행됩니다.
    """

    # 기본 작업 디렉토리 설정
    BASE_PATH = r"C:\Users\dbk32\OneDrive\바탕 화면\test" 

    # 파이프라인 객체 생성
    pipeline = WindPowerPipeline(base_path=BASE_PATH, max_workers=8)

    print("=== 디렉토리 구조 확인 ===")
    print(f"기본 경로: {pipeline.base_path}")
    print(f"LDAPS 데이터: {pipeline.ldaps_path}")
    print(f"타겟 데이터: {pipeline.target_path}")
    print(f"결과 저장: {pipeline.result_path}")

    # 필요한 디렉토리가 존재하는지 확인
    required_paths = {
        'ldaps': pipeline.ldaps_path,
        'target': pipeline.target_path,
        '파생변수 데이터': pipeline.derived_data_path
    }

    missing_paths = []
    for name, path in required_paths.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (없음)")
            missing_paths.append(name)

    if missing_paths:
        print(f"\n경고: 다음 경로가 없습니다: {', '.join(missing_paths)}")
        print("파생변수 데이터가 없으면 전체 전처리부터 시작합니다.")

    try:
        # 전체 파이프라인 실행
        # use_processed_data=True: 이미 파생변수가 추가된 gy.parquet, yy.parquet, yd.parquet 사용
        # skip_weather_processing=True: 이미 전처리된 MAX 데이터 사용
        # skip_weather_processing=False: 처음부터 날씨 데이터 전처리 시작

        # 우선순위:
        # 1순위 - 최종 처리된 데이터 (gy.parquet, yy.parquet, yd.parquet)
        # 2순위 - MAX 데이터에서 파생변수 생성
        # 3순위 - 원본 데이터에서 전체 전처리

        use_processed = pipeline.derived_data_path.exists()
        skip_weather = not use_processed and any([
            (pipeline.result_path / "경주_test/경주_timeseries_MAX.parquet").exists(),
            (pipeline.result_path / "양양_test/양양_timeseries_MAX.parquet").exists(),
            (pipeline.result_path / "영덕_test/영덕_timeseries_MAX.parquet").exists()
        ])

        final_result = pipeline.run_full_pipeline(
            skip_weather_processing=skip_weather,
            use_processed_data=use_processed
        )

        print("\n=== 최종 결과 요약 ===")
        print(f"총 예측 결과: {len(final_result)}개")
        print("\n지역별 예측 개수:")
        for plant in final_result['plant_name'].unique():
            count = len(final_result[final_result['plant_name'] == plant])
            print(f"  {plant}: {count}개")

        print(f"\n결과 파일: {pipeline.result_path / 'result.csv'}")

    except Exception as e:
        print(f"파이프라인 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

# 기본 작업 디렉토리 설정
BASE_PATH = r"C:\Users\dbk32\OneDrive\바탕 화면\test"

## Case 1
# pipeline = WindPowerPipeline(base_path=BASE_PATH, max_workers=8)
#
# # 전체 파이프라인 실행 (모든 단계 포함)
# final_result = pipeline.run_full_pipeline(
#     skip_weather_processing=False,  # 원본부터 전처리
#     use_processed_data=False        # 중간 데이터 사용 안함
# )


## Case 2
# pipeline = WindPowerPipeline(base_path=BASE_PATH)
# final_result = pipeline.run_full_pipeline(
#     skip_weather_processing=True,   # MAX 데이터 사용
#     use_processed_data=False
# )

## Case 3

pipeline = WindPowerPipeline(base_path=BASE_PATH, max_workers=8)

# 최종 데이터로 바로 모델링
final_result = pipeline.run_full_pipeline(
    use_processed_data=True  # 최종 데이터 직접 사용
)
