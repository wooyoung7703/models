"""
XGBoost 모델 사용 예제
실제 암호화폐 데이터를 사용한 훈련 및 예측 예제
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'app'))

from training.train_xgboost import XGBoostTrainer, create_sample_data
from models.predict_xgboost import XGBoostPredictor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    
    print("="*70)
    print("XGBoost 암호화폐 가격 예측 시스템")
    print("="*70)
    print()
    
    # 1. 데이터 생성 (실제로는 데이터베이스에서 가져옴)
    logger.info("1단계: 암호화폐 1분봉 데이터 생성...")
    data = create_sample_data(n_samples=10000)  # 10000개의 1분봉 데이터
    logger.info(f"   - 생성된 데이터: {len(data)}개 행")
    logger.info(f"   - 기간: {data['timestamp'].min()} ~ {data['timestamp'].max()}")
    print()
    
    # 2. 피처 선택
    logger.info("2단계: 피처 선택...")
    feature_cols = [col for col in data.columns if col not in ['timestamp', 'low']]
    logger.info(f"   - 선택된 피처: {', '.join(feature_cols)}")
    logger.info(f"   - 타겟 변수: low (최저가)")
    print()
    
    # 3. 모델 훈련
    logger.info("3단계: XGBoost 모델 훈련...")
    trainer = XGBoostTrainer(model_path="backend/app/models/xgboost")
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        data,
        target_column='low',
        feature_columns=feature_cols,
        test_size=0.2
    )
    
    logger.info("   모델 훈련 시작...")
    model = trainer.train(X_train, y_train, X_test, y_test)
    logger.info("   ✓ 모델 훈련 완료")
    print()
    
    # 4. 모델 저장
    logger.info("4단계: 모델 저장...")
    model_path = trainer.save_model()
    logger.info(f"   ✓ 모델 저장 완료: {model_path}")
    print()
    
    # 5. 예측 수행
    logger.info("5단계: 예측 수행...")
    predictor = XGBoostPredictor(model_path="backend/app/models/xgboost")
    predictor.load_model()
    
    # 테스트 데이터로 예측
    predictions = predictor.predict_batch(X_test[:20])
    actuals = y_test[:20]
    
    print()
    print("예측 결과 (처음 20개):")
    print("-" * 70)
    print(f"{'번호':<5} {'실제값':>15} {'예측값':>15} {'오차':>15} {'오차율(%)':>12}")
    print("-" * 70)
    
    total_error = 0
    for i, (pred, actual) in enumerate(zip(predictions, actuals), 1):
        error = abs(pred - actual)
        error_pct = (error / actual) * 100
        total_error += error
        print(f"{i:<5} {actual:>15.2f} {pred:>15.2f} {error:>15.2f} {error_pct:>11.3f}%")
    
    print("-" * 70)
    avg_error = total_error / len(predictions)
    avg_error_pct = (avg_error / actuals.mean()) * 100
    print(f"평균 오차: {avg_error:.2f} ({avg_error_pct:.3f}%)")
    print()
    
    # 6. 신뢰 구간과 함께 예측
    logger.info("6단계: 신뢰 구간과 함께 예측...")
    predictions, lower, upper = predictor.predict_with_confidence(
        X_test[:10], 
        return_intervals=True
    )
    
    print()
    print("신뢰 구간 예측 결과 (처음 10개):")
    print("-" * 90)
    print(f"{'번호':<5} {'예측값':>15} {'하한':>15} {'상한':>15} {'실제값':>15} {'구간내':>8}")
    print("-" * 90)
    
    in_interval_count = 0
    for i, (pred, low, up, actual) in enumerate(zip(predictions, lower, upper, actuals[:10]), 1):
        in_interval = low <= actual <= up
        if in_interval:
            in_interval_count += 1
        status = "✓" if in_interval else "✗"
        print(f"{i:<5} {pred:>15.2f} {low:>15.2f} {up:>15.2f} {actual:>15.2f} {status:>8}")
    
    print("-" * 90)
    print(f"신뢰 구간 내 예측: {in_interval_count}/10 ({in_interval_count*10}%)")
    print()
    
    # 7. 최종 요약
    print("="*70)
    print("요약")
    print("="*70)
    print(f"✓ 훈련 데이터: {len(X_train)}개")
    print(f"✓ 테스트 데이터: {len(X_test)}개")
    print(f"✓ 피처 개수: {len(feature_cols)}개")
    print(f"✓ 평균 예측 오차: {avg_error:.2f} ({avg_error_pct:.3f}%)")
    print(f"✓ 모델 저장 위치: {model_path}")
    print()
    print("🎉 XGBoost 모델이 정상적으로 예측을 수행할 수 있습니다!")
    print("="*70)


if __name__ == "__main__":
    main()
