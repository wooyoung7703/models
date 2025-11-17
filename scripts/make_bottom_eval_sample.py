import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_sample(n_rows: int, horizon: int) -> pd.DataFrame:
    # 시간 인덱스 (단순한 정수 시퀀스)
    ts = np.arange(n_rows)

    # 기본 가격 시뮬레이션 (랜덤 워크)
    prices = 10000 + np.cumsum(np.random.randn(n_rows) * 50)

    # 저점 레이블: 랜덤하게 몇 %만 1로
    bottom_label = (np.random.rand(n_rows) < 0.07).astype(int)

    # 저점 확률: 레이블과 약간 상관 있게 생성
    base_prob = np.clip(
        0.1 * np.random.rand(n_rows) + 0.6 * bottom_label + 0.2 * np.random.rand(n_rows),
        0.0,
        1.0,
    )

    data = {
        "timestamp": ts,
        "close": prices,
        "bottom_label": bottom_label,
        "bottom_prob": base_prob,
    }

    # 예측 경로: 현재 가격에 약간의 드리프트와 노이즈를 더해서 생성
    for k in range(1, horizon + 1):
        drift = (np.random.randn(n_rows) * 10) + (k * np.random.randn(n_rows))
        data[f"fwd_pred_{k}"] = prices + drift

    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="bottom_vs_forecast 분석용 샘플 CSV 생성")
    parser.add_argument("--out", type=str, required=True, help="생성할 CSV 경로 (예: data/bottom_eval_sample.csv)")
    parser.add_argument("--rows", type=int, default=1000, help="샘플 행 수 (기본 1000)")
    parser.add_argument("--horizon", type=int, default=10, help="예측 horizon (fwd_pred_1..H, 기본 10)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = make_sample(args.rows, args.horizon)
    df.to_csv(out_path, index=False)
    print(f"샘플 CSV를 '{out_path}' 에 생성했습니다. 행 수={len(df)}, horizon={args.horizon}")


if __name__ == "__main__":
    main()
