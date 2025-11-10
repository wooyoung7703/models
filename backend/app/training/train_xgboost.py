"""
XGBoost 모델 훈련 모듈
암호화폐 가격 예측을 위한 XGBoost 모델 훈련
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import xgboost as xgb
import joblib
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """XGBoost 모델 훈련을 위한 클래스"""
    
    def __init__(self, model_path: str = "models/xgboost"):
        """
        Args:
            model_path: 모델을 저장할 경로
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        
        # 모델 저장 디렉토리 생성
        os.makedirs(model_path, exist_ok=True)
        
    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'low',
        feature_columns: Optional[list] = None,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터 전처리 및 훈련/테스트 세트 분리
        
        Args:
            data: 입력 데이터프레임
            target_column: 예측할 타겟 컬럼명
            feature_columns: 사용할 피처 컬럼 리스트 (None이면 타겟 제외 모든 컬럼)
            test_size: 테스트 세트 비율
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"데이터 전처리 시작 - 전체 데이터 크기: {len(data)} 행")
        
        # 피처와 타겟 분리
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        self.feature_names = feature_columns
        logger.info(f"사용할 피처 개수: {len(feature_columns)}")
        logger.info(f"피처 목록: {feature_columns}")
        
        X = data[feature_columns].values
        y = data[target_column].values
        
        # 훈련/테스트 분리
        split_idx = int(len(data) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"훈련 데이터: {len(X_train)} 행")
        logger.info(f"테스트 데이터: {len(X_test)} 행")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> xgb.Booster:
        """
        XGBoost 모델 훈련
        
        Args:
            X_train: 훈련 피처
            y_train: 훈련 타겟
            X_val: 검증 피처 (옵션)
            y_val: 검증 타겟 (옵션)
            params: XGBoost 파라미터 (옵션)
            
        Returns:
            훈련된 XGBoost 모델
        """
        logger.info("XGBoost 모델 훈련 시작")
        logger.info(f"훈련 데이터 크기: {X_train.shape}")
        
        # 기본 파라미터 설정
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        logger.info(f"모델 파라미터: {params}")
        
        # DMatrix 생성
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        # 검증 데이터가 있는 경우
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'val'))
            logger.info(f"검증 데이터 크기: {X_val.shape}")
        
        # 모델 훈련
        num_boost_round = params.pop('n_estimators', 100)
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            verbose_eval=10
        )
        
        logger.info("모델 훈련 완료")
        return self.model
    
    def save_model(self, filename: str = "xgboost_model.json") -> str:
        """
        모델 저장
        
        Args:
            filename: 저장할 파일명
            
        Returns:
            저장된 모델 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. 먼저 모델을 훈련하세요.")
        
        model_file = os.path.join(self.model_path, filename)
        self.model.save_model(model_file)
        
        # 피처 이름도 함께 저장
        feature_file = os.path.join(self.model_path, "feature_names.pkl")
        joblib.dump(self.feature_names, feature_file)
        
        logger.info(f"모델 저장 완료: {model_file}")
        logger.info(f"피처 이름 저장 완료: {feature_file}")
        
        return model_file
    
    def load_model(self, filename: str = "xgboost_model.json") -> xgb.Booster:
        """
        저장된 모델 로드
        
        Args:
            filename: 로드할 파일명
            
        Returns:
            로드된 XGBoost 모델
        """
        model_file = os.path.join(self.model_path, filename)
        feature_file = os.path.join(self.model_path, "feature_names.pkl")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file}")
        
        self.model = xgb.Booster()
        self.model.load_model(model_file)
        
        # 피처 이름 로드
        if os.path.exists(feature_file):
            self.feature_names = joblib.load(feature_file)
        
        logger.info(f"모델 로드 완료: {model_file}")
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X: 예측할 피처 데이터
            
        Returns:
            예측 결과
        """
        if self.model is None:
            raise ValueError("예측할 모델이 없습니다. 먼저 모델을 훈련하거나 로드하세요.")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dtest)
        
        logger.info(f"예측 완료 - 데이터 크기: {X.shape}, 예측 결과 크기: {predictions.shape}")
        
        return predictions


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    테스트용 샘플 데이터 생성 (암호화폐 가격 시뮬레이션)
    
    Args:
        n_samples: 생성할 샘플 개수
        
    Returns:
        샘플 데이터프레임
    """
    np.random.seed(42)
    
    # 시간 인덱스 생성
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
    
    # 기본 가격 생성 (랜덤 워크)
    base_price = 50000
    returns = np.random.randn(n_samples) * 100
    prices = base_price + np.cumsum(returns)
    
    # OHLC 데이터 생성
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices + np.random.randn(n_samples) * 50,
        'high': prices + np.abs(np.random.randn(n_samples)) * 100,
        'low': prices - np.abs(np.random.randn(n_samples)) * 100,
        'close': prices + np.random.randn(n_samples) * 50,
        'volume': np.random.randint(100, 10000, n_samples)
    })
    
    # 기술적 지표 추가
    data['price_range'] = data['high'] - data['low']
    data['price_change'] = data['close'] - data['open']
    data['ma_5'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['ma_20'] = data['close'].rolling(window=20, min_periods=1).mean()
    data['volatility'] = data['close'].rolling(window=20, min_periods=1).std()
    
    # NaN 제거
    data = data.fillna(method='bfill')
    
    logger.info(f"샘플 데이터 생성 완료: {len(data)} 행, {len(data.columns)} 열")
    
    return data


if __name__ == "__main__":
    # 예제 실행
    logger.info("="*50)
    logger.info("XGBoost 모델 훈련 예제 시작")
    logger.info("="*50)
    
    # 샘플 데이터 생성
    data = create_sample_data(n_samples=5000)
    
    # 훈련에 사용할 피처 선택 (timestamp 제외)
    feature_cols = [col for col in data.columns if col not in ['timestamp', 'low']]
    
    # 트레이너 초기화
    trainer = XGBoostTrainer()
    
    # 데이터 준비
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        data,
        target_column='low',
        feature_columns=feature_cols
    )
    
    # 모델 훈련
    model = trainer.train(X_train, y_train, X_test, y_test)
    
    # 모델 저장
    trainer.save_model()
    
    # 예측 테스트
    logger.info("="*50)
    logger.info("예측 테스트")
    logger.info("="*50)
    
    predictions = trainer.predict(X_test[:10])
    logger.info(f"실제값: {y_test[:10]}")
    logger.info(f"예측값: {predictions[:10]}")
    
    # 모델 로드 테스트
    logger.info("="*50)
    logger.info("모델 로드 및 재예측 테스트")
    logger.info("="*50)
    
    new_trainer = XGBoostTrainer()
    new_trainer.load_model()
    new_predictions = new_trainer.predict(X_test[:10])
    logger.info(f"재예측값: {new_predictions[:10]}")
    
    logger.info("="*50)
    logger.info("XGBoost 모델 훈련 및 예측 테스트 완료!")
    logger.info("="*50)
