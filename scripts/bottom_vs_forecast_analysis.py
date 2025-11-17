import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    # 선택 의존성: 없으면 메타 모델 학습은 건너뛰고 요약만 수행
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - 런타임 환경에 따라 다를 수 있음
    LogisticRegression = None


@dataclass
class BottomVsForecastConfig:
    label_col: str = "bottom_label"  # 0/1 bottom label
    prob_col: str = "bottom_prob"    # model probability at time t
    price_col: str = "close"         # actual price at time t
    forecast_prefix: str = "fwd_pred_"  # columns like fwd_pred_1, fwd_pred_2, ...


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def infer_forecast_horizon(df: pd.DataFrame, prefix: str) -> int:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No forecast columns found with prefix '{prefix}'")
    horizons = []
    for c in cols:
        try:
            horizons.append(int(c[len(prefix):]))
        except ValueError:
            continue
    if not horizons:
        raise ValueError(f"Forecast columns with prefix '{prefix}' must end with integer horizon, e.g. {prefix}1, {prefix}2")
    return max(horizons)


def compute_forecast_features(df: pd.DataFrame, cfg: BottomVsForecastConfig) -> pd.DataFrame:
    h = infer_forecast_horizon(df, cfg.forecast_prefix)
    forecast_cols = [f"{cfg.forecast_prefix}{k}" for k in range(1, h + 1)]
    for c in forecast_cols:
        if c not in df.columns:
            raise ValueError(f"Missing forecast column '{c}' in input frame")

    forecast_arr = df[forecast_cols].to_numpy(dtype=float)
    # 예측 경로 기준 최소/최대/평균
    f_min = forecast_arr.min(axis=1)
    f_max = forecast_arr.max(axis=1)
    f_mean = forecast_arr.mean(axis=1)

    price = df[cfg.price_col].to_numpy(dtype=float)

    # 현재 가격 대비 예측 최저가까지의 하락 여유 (음수면 이미 예측 최저가보다 아래)
    room_to_forecast_min = (f_min - price) / price

    # 예측 평균 대비 현재 위치 (0이면 평균, 음수면 평균보다 아래)
    rel_to_forecast_mean = (price - f_mean) / f_mean

    # 예측 경로의 단순 expected return (평균 / 현재 - 1)
    forecast_expected_return = f_mean / price - 1.0

    out = df.copy()
    out["f_min"] = f_min
    out["f_max"] = f_max
    out["f_mean"] = f_mean
    out["room_to_forecast_min"] = room_to_forecast_min
    out["rel_to_forecast_mean"] = rel_to_forecast_mean
    out["forecast_expected_return"] = forecast_expected_return

    return out


def bin_by_prob(prob: np.ndarray, n_bins: int = 10) -> np.ndarray:
    # 균등 확률 구간으로 binning
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.digitize(prob, edges, right=True)


def summarize(df: pd.DataFrame, cfg: BottomVsForecastConfig, n_bins: int = 10) -> pd.DataFrame:
    if cfg.label_col not in df.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found")
    if cfg.prob_col not in df.columns:
        raise ValueError(f"Prob column '{cfg.prob_col}' not found")

    df = df.dropna(subset=[cfg.label_col, cfg.prob_col]).copy()

    df["prob_bin"] = bin_by_prob(df[cfg.prob_col].to_numpy(dtype=float), n_bins=n_bins)

    # 각 bin(저점 확률 구간)에서 실제 바닥 발생률, 예측 대비 저점 위치를 요약
    grouped = df.groupby("prob_bin")

    rows = []
    for bin_id, g in grouped:
        if g.empty:
            continue
        p_min = g[cfg.prob_col].min()
        p_max = g[cfg.prob_col].max()
        p_mean = g[cfg.prob_col].mean()
        bottom_rate = g[cfg.label_col].mean()

        room_mean = g["room_to_forecast_min"].mean() if "room_to_forecast_min" in g.columns else np.nan
        rel_mean = g["rel_to_forecast_mean"].mean() if "rel_to_forecast_mean" in g.columns else np.nan
        f_exp_mean = g["forecast_expected_return"].mean() if "forecast_expected_return" in g.columns else np.nan

        rows.append(
            {
                "prob_bin": bin_id,
                "prob_min": p_min,
                "prob_max": p_max,
                "prob_mean": p_mean,
                "bottom_rate": bottom_rate,
                "room_to_forecast_min_mean": room_mean,
                "rel_to_forecast_mean_mean": rel_mean,
                "forecast_expected_return_mean": f_exp_mean,
                "count": len(g),
            }
        )

    return pd.DataFrame(rows).sort_values("prob_bin").reset_index(drop=True)


def fit_meta_model(df: pd.DataFrame, cfg: BottomVsForecastConfig) -> Optional[dict]:
    if LogisticRegression is None:
        print("sklearn 이 없어서 메타 로지스틱 모델 학습은 건너뜁니다.")
        return None

    required_cols = [
        cfg.label_col,
        cfg.prob_col,
        "room_to_forecast_min",
        "rel_to_forecast_mean",
        "forecast_expected_return",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"메타 모델 학습에 필요한 컬럼이 없습니다: {c}")

    work = df.dropna(subset=required_cols).copy()
    if work.empty:
        print("메타 모델 학습에 사용 가능한 행이 없습니다.")
        return None

    # 특징: 기존 bottom_prob + 예측 대비 위치 요약 3개
    X = work[[
        cfg.prob_col,
        "room_to_forecast_min",
        "rel_to_forecast_mean",
        "forecast_expected_return",
    ]].to_numpy(dtype=float)
    y = work[cfg.label_col].to_numpy(dtype=int)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    coefs = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])

    # 간단한 출력 및 구조화된 결과 반환
    print("\n=== 예측 대비 저점 메타 로지스틱 모델 ===")
    print("특징 순서: [bottom_prob, room_to_forecast_min, rel_to_forecast_mean, forecast_expected_return]")
    print("계수(coef):", coefs)
    print("절편(intercept):", f"{intercept:0.6f}")

    return {
        "features": [
            cfg.prob_col,
            "room_to_forecast_min",
            "rel_to_forecast_mean",
            "forecast_expected_return",
        ],
        "coef": coefs,
        "intercept": intercept,
    }


def run(input_csv: Path, output_csv: Optional[Path], meta_json: Optional[Path]) -> None:
    cfg = BottomVsForecastConfig()

    df = load_frame(input_csv)
    df = compute_forecast_features(df, cfg)
    summary = summarize(df, cfg)

    print("=== 저점 확률 구간별 요약 ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False)
        print(f"\n요약 결과를 '{output_csv}' 에 저장했습니다.")

    # 선택: 메타 로지스틱 모델 학습 및 저장
    if meta_json is not None:
        import json

        meta = fit_meta_model(df, cfg)
        if meta is not None:
            meta_json.parent.mkdir(parents=True, exist_ok=True)
            with meta_json.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"메타 로지스틱 모델을 '{meta_json}' 에 저장했습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="미리 예측한 가격 대비 저점 스코어 분석")
    parser.add_argument("input_csv", type=str, help="시계열 데이터 CSV (bottom_prob, bottom_label, close, fwd_pred_* 포함)")
    parser.add_argument("--out", dest="output_csv", type=str, default=None, help="요약 결과를 저장할 CSV 경로 (옵션)")
    parser.add_argument("--meta-json", dest="meta_json", type=str, default=None, help="예측 대비 저점 메타 로지스틱 계수를 저장할 json 경로 (옵션)")

    args = parser.parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv is not None else None
    meta_path = Path(args.meta_json) if args.meta_json is not None else None

    run(input_path, output_path, meta_path)
