"""
XGBoost 모델 예측 모듈
훈련된 모델을 사용한 예측 기능
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Optional
import os
import sys

# 현재 파일의 디렉토리를 기준으로 상위 디렉토리 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from training.train_xgboost import XGBoostTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost 모델 예측을 위한 클래스"""
    
    def __init__(self, model_path: str = "models/xgboost"):
        """
        Args:
            model_path: 모델이 저장된 경로
        """
        self.model_path = model_path
        self.trainer = XGBoostTrainer(model_path=model_path)
        self.model_loaded = False
        
    def load_model(self, model_filename: str = "xgboost_model.json") -> None:
        """
        저장된 모델 로드
        
        Args:
            model_filename: 로드할 모델 파일명
        """
        try:
            self.trainer.load_model(model_filename)
            self.model_loaded = True
            logger.info("모델 로드 성공 - 예측 준비 완료")
        except FileNotFoundError as e:
            logger.error(f"모델 파일을 찾을 수 없습니다: {e}")
            raise
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    def predict_single(self, features: Union[dict, np.ndarray]) -> float:
        """
        단일 데이터 포인트 예측
        
        Args:
            features: 예측할 피처 (딕셔너리 또는 numpy 배열)
            
        Returns:
            예측된 low 가격
        """
        if not self.model_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. 먼저 load_model()을 호출하세요.")
        
        # 딕셔너리를 numpy 배열로 변환
        if isinstance(features, dict):
            # 피처 순서에 맞게 배열 생성
            feature_array = np.array([features[name] for name in self.trainer.feature_names])
            feature_array = feature_array.reshape(1, -1)
        else:
            if features.ndim == 1:
                feature_array = features.reshape(1, -1)
            else:
                feature_array = features
        
        prediction = self.trainer.predict(feature_array)
        return float(prediction[0])
    
    def predict_batch(
        self, 
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        배치 데이터 예측
        
        Args:
            features: 예측할 피처들 (데이터프레임 또는 numpy 배열)
            
        Returns:
            예측 결과 배열
        """
        if not self.model_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. 먼저 load_model()을 호출하세요.")
        
        # 데이터프레임을 numpy 배열로 변환
        if isinstance(features, pd.DataFrame):
            # 피처 이름 순서에 맞게 선택
            features_array = features[self.trainer.feature_names].values
        else:
            features_array = features
        
        predictions = self.trainer.predict(features_array)
        logger.info(f"배치 예측 완료 - {len(predictions)}개 데이터 포인트")
        
        return predictions
    
    def predict_with_confidence(
        self, 
        features: Union[pd.DataFrame, np.ndarray],
        return_intervals: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        예측값과 함께 신뢰도 정보 반환
        
        Args:
            features: 예측할 피처들
            return_intervals: True이면 예측 구간도 함께 반환
            
        Returns:
            예측값 또는 (예측값, 하한, 상한) 튜플
        """
        predictions = self.predict_batch(features)
        
        if return_intervals:
            # 간단한 예측 구간 추정 (표준편차 기반)
            std_dev = np.std(predictions) if len(predictions) > 1 else 0
            lower_bound = predictions - 1.96 * std_dev
            upper_bound = predictions + 1.96 * std_dev
            return predictions, lower_bound, upper_bound
        
        return predictions


def get_latest_data_for_prediction(n_points: int = 1) -> pd.DataFrame:
    """
    예측을 위한 최신 데이터 가져오기 (데모용 함수)
    실제로는 데이터베이스나 API에서 데이터를 가져와야 함
    
    Args:
        n_points: 가져올 데이터 포인트 수
        
    Returns:
        최신 데이터 데이터프레임
    """
    # 데모용 샘플 데이터 생성
    from training.train_xgboost import create_sample_data
    
    full_data = create_sample_data(n_samples=100)
    latest_data = full_data.tail(n_points)
    
    logger.info(f"최신 데이터 {n_points}개 포인트 가져오기 완료")
    return latest_data


if __name__ == "__main__":
    logger.info("="*50)
    logger.info("XGBoost 예측 모듈 테스트")
    logger.info("="*50)
    
    # 예측기 초기화
    predictor = XGBoostPredictor()
    
    # 모델 로드
    try:
        predictor.load_model()
    except FileNotFoundError:
        logger.warning("모델 파일이 없습니다. 먼저 train_xgboost.py를 실행하여 모델을 훈련하세요.")
        logger.info("데모용 모델 훈련을 시작합니다...")
        
        # 데모용 훈련
        from training.train_xgboost import create_sample_data
        
        data = create_sample_data(n_samples=1000)
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'low']]
        
        trainer = XGBoostTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            data,
            target_column='low',
            feature_columns=feature_cols
        )
        trainer.train(X_train, y_train)
        trainer.save_model()
        
        # 모델 다시 로드
        predictor.load_model()
    
    # 최신 데이터 가져오기
    latest_data = get_latest_data_for_prediction(n_points=5)
    
    # 피처 준비
    feature_cols = [col for col in latest_data.columns if col not in ['timestamp', 'low']]
    features = latest_data[feature_cols]
    
    logger.info("="*50)
    logger.info("단일 예측 테스트")
    logger.info("="*50)
    
    # 단일 예측
    first_point = features.iloc[0].to_dict()
    prediction = predictor.predict_single(first_point)
    actual = latest_data.iloc[0]['low']
    
    logger.info(f"예측값: {prediction:.2f}")
    logger.info(f"실제값: {actual:.2f}")
    logger.info(f"오차: {abs(prediction - actual):.2f}")
    
    logger.info("="*50)
    logger.info("배치 예측 테스트")
    logger.info("="*50)
    
    # 배치 예측
    batch_predictions = predictor.predict_batch(features)
    actuals = latest_data['low'].values
    
    for i, (pred, actual) in enumerate(zip(batch_predictions, actuals)):
        logger.info(f"데이터 {i+1}: 예측={pred:.2f}, 실제={actual:.2f}, 오차={abs(pred-actual):.2f}")
    
    logger.info("="*50)
    logger.info("신뢰 구간 예측 테스트")
    logger.info("="*50)
    
    # 신뢰 구간과 함께 예측
    predictions, lower, upper = predictor.predict_with_confidence(features, return_intervals=True)
    
    for i, (pred, low, up, actual) in enumerate(zip(predictions, lower, upper, actuals)):
        in_interval = low <= actual <= up
        logger.info(
            f"데이터 {i+1}: 예측={pred:.2f}, "
            f"구간=[{low:.2f}, {up:.2f}], "
            f"실제={actual:.2f}, "
            f"구간내={in_interval}"
        )
    
    logger.info("="*50)
    logger.info("XGBoost 예측 모듈 테스트 완료!")
    logger.info("모델이 정상적으로 예측을 수행할 수 있습니다.")
    logger.info("="*50)
