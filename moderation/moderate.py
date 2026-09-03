# -*- coding: utf-8 -*-
"""سیستم تشخیص کامنت نامناسب فارسی — نسخه ۳ (الزام: رگرسیون لجستیک)

آستانه‌ها از model_config.json خوانده می‌شوند (انتخاب‌شده روی validation جداگانه —
نه روی تست‌ها). اگر فایل نبود، مقادیر پیش‌فرض محافظه‌کارانه.

معماری تصمیم:
    ok      → انتشار فوری
    review  → صف بازبینی انسانی     (p2 >= review_threshold)
    block   → مسدودسازی خودکار      (p1 >= block_v1 و p2 >= block_v2)

هر دو مدل LogisticRegression هستند:
    model.joblib     ← v2: پهن، ۵ منبع (۱۰۹,۷۴۷)
    model_v1.joblib  ← v1: سخت‌گیر فحش (توییتر، ۲۶,۶۱۶)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import joblib  # noqa: E402
from mod_text import normalize  # noqa: E402

DEFAULTS = {'review_threshold': 0.448, 'block_v1': 0.8, 'block_v2': 0.9}
_models = {}
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


def _load(name):
    if name not in _models:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{name} پیدا نشد — باید کنار moderate.py باشد')
        _models[name] = joblib.load(path)
    return _models[name]


def predict(texts, review_threshold=None):
    """برای هر متن: احتمال نامناسب بودن (v2) + تصمیم ok/review/block."""
    cfg = _config()
    if review_threshold is None:
        review_threshold = cfg['review_threshold']
    v2 = _load('model.joblib')
    v1 = _load('model_v1.joblib')
    norm = [normalize(t) for t in texts]
    p2 = v2.predict_proba(norm)[:, 1]
    p1 = v1.predict_proba(norm)[:, 1]
    out = []
    for t, a, b in zip(texts, p2, p1):
        if b >= cfg['block_v1'] and a >= cfg['block_v2']:
            decision = 'block'
        elif a >= review_threshold:
            decision = 'review'
        else:
            decision = 'ok'
        out.append({
            'text': t,
            'prob_abusive': round(float(a), 4),
            'prob_v1': round(float(b), 4),
            'decision': decision,
        })
    return out


def reload_models():
    """پاک‌کردن کش مدل و پیکربندی (بعد از آموزش/بازگردانی)."""
    _models.clear()
    global _cfg
    _cfg = None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='تشخیص کامنت نامناسب فارسی (ok/review/block)')
    ap.add_argument('texts', nargs='+', help='متن کامنت(ها)')
    args = ap.parse_args()
    icons = {'ok': '✅ ok', 'review': '🔍 بازبینی', 'block': '🚫 بلاک'}
    for r in predict(args.texts):
        print(f"{icons[r['decision']]:<12s} (p={r['prob_abusive']:.2f})  {r['text'][:80]}")
