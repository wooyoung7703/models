"""Quick diagnostic for XGB feature mapping against Candle schema.

Run inside backend environment:
    python scripts/xgb_feature_health.py

Outputs counts for: total, matched Candle fields, generic (f0,f1,..) placeholders, missing.
"""
from importlib import import_module
from pprint import pprint

try:
    from backend.app.model_adapters import XGBTabularAdapter
    from backend.app.models import Candle
except ImportError:
    # Allow running from repo root with PYTHONPATH adjustment suggestion
    raise SystemExit("Import failed. Run as: python -m backend.app.model_adapters or set PYTHONPATH to repo root.")

# Collect candle field names (pydantic v1/v2 compatibility)
fields = []
for attr in ('model_fields','__fields__','__annotations__'):
    obj = getattr(Candle, attr, None)
    if isinstance(obj, dict):
        fields = list(obj.keys())
        break
candle_set = set(fields)

adapter = XGBTabularAdapter()
features = getattr(adapter, '_features', [])
if not features:
    print('[xgb_health] No feature list present; model may be raw or not found.')
    raise SystemExit(0)

generic = [f for f in features if f.startswith('f') and f[1:].isdigit()]
matched = [f for f in features if f in candle_set]
missing = [f for f in features if f not in candle_set and f not in generic]

print('[xgb_health] Summary')
print(f'  total:   {len(features)}')
print(f'  matched: {len(matched)}')
print(f'  generic: {len(generic)}')
print(f'  missing: {len(missing)}')
if missing:
    print('  missing_first_20:', missing[:20])
if matched:
    # Show a sample of matched features for manual verification
    print('  matched_sample:', matched[:20])
