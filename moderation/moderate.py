# -*- coding: utf-8 -*-
"""سیستم تشخیص کامنت نامناسب فارسی — نسخه ۴ (تک‌مدل، بدون بازبینی).

تصمیم فقط دوحالته:
    p ≥ block_threshold → block (نامناسب)
    در غیر این صورت     → ok (مجاز)

آستانه از model_config.json (انتخاب‌شده روی validation). مدل: LogisticRegression.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import joblib  # noqa: E402
from mod_text import normalize  # noqa: E402

DEFAULTS = {'block_threshold': 0.509}
_model = None
_cfg = None


def _config():
    global _cfg
    if _cfg is None:
        try:
            with open(os.path.join(HERE, 'model_config.json'), encoding='utf-8') as f:
                _cfg = {**DEFAULTS, **json.load(f)}
        except Exception:
            _cfg = dict(DEFAULTS)
    return _cfg


def _load():
    global _model
    if _model is None:
        path = os.path.join(HERE, 'model.joblib')
        if not os.path.exists(path):
            raise FileNotFoundError('model.joblib پیدا نشد — باید کنار moderate.py باشد')
        _model = joblib.load(path)
    return _model


def predict(texts, threshold=None):
    """برای هر متن: احتمال نامناسب بودن + تصمیم ok / block."""
    if threshold is None:
        threshold = _config()['block_threshold']
    model = _load()
    probs = model.predict_proba([normalize(t) for t in texts])[:, 1]
    return [
        {'text': t,
         'prob_abusive': round(float(p), 4),
         'decision': 'block' if p >= threshold else 'ok',
         'threshold': threshold}
        for t, p in zip(texts, probs)
    ]


def reload_models():
    """پاک‌کردن کش (بعد از آموزش/بازگردانی)."""
    global _model, _cfg
    _model = None
    _cfg = None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='تشخیص کامنت نامناسب فارسی (ok/block)')
    ap.add_argument('texts', nargs='+', help='متن کامنت(ها)')
    args = ap.parse_args()
    icons = {'ok': '✅ مجاز', 'block': '🚫 بلاک'}
    for r in predict(args.texts):
        print(f"{icons[r['decision']]:<10s} (p={r['prob_abusive']:.2f})  {r['text'][:80]}")
