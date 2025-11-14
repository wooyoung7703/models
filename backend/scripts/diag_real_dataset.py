import json
from backend.app.training.dataset_real import build_bottom_tabular_dataset_real

def main():
    try:
        X,y,feat = build_bottom_tabular_dataset_real(days=720, interval='1m', past_window=15, future_window=60, tolerance_pct=0.004)
        print(json.dumps({
            'ok': True,
            'X_shape': [int(X.shape[0]), int(X.shape[1])],
            'y_len': int(len(y)),
            'y_pos': int((y==1).sum()),
            'feat_cnt': int(len(feat)),
        }))
    except Exception as e:
        import traceback
        print(json.dumps({'ok': False, 'err': str(e)}))
        traceback.print_exc()

if __name__ == '__main__':
    main()
