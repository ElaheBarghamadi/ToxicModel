# -*- coding: utf-8 -*-
"""سیستم تشخیص کامنت نامناسب فارسی — نسخه ۲ (دو-مدلی)

معماری تصمیم:
    ok      → انتشار فوری
    review  → صف بازبینی انسانی     (v2 >= 0.50)
    block   → مسدودسازی خودکار      (v1 >= 0.90 و v2 >= 0.70)

- v2 (model.joblib):      آموزش‌دیده روی ۱۰۹,۷۴۷ نمونه از ۵ منبع (توییتر، تلگرام،
                          اینستاگرام×۲، PHate) — پوشش توهین ملایم + فحش
- v1 (model_v1.joblib):   مدل سخت‌گیر فحش (فقط توییتر) — دقت ۹۹٫۸٪ در آستانه ۰٫۹

استفاده:
    from moderate import predict
    for r in predict(["متن کامنت"]):
        print(r["decision"], r["prob_abusive"])
خط فرمان:
    python moderate.py "متن کامتن ۱" "متن ۲"
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import joblib  # noqa: E402
from mod_text import normalize  # noqa: E402

REVIEW_THRESHOLD = 0.5      # v2: صف بازبینی
BLOCK_V1 = 0.9              # v1: اطمینان بالا
BLOCK_V2 = 0.7              # v2: تأیید دوم
_models = {}


def _load(name):
    if name not in _models:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} پیدا نشد — باید کنار moderate.py باشد")
        _models[name] = joblib.load(path)
    return _models[name]


def predict(texts, review_threshold=REVIEW_THRESHOLD):
    """برای هر متن: احتمال نامناسب بودن (v2) + تصمیم ok/review/block."""
    v2 = _load("model.joblib")
    v1 = _load("model_v1.joblib")
    norm = [normalize(t) for t in texts]
    p2 = v2.predict_proba(norm)[:, 1]
    p1 = v1.predict_proba(norm)[:, 1]
    out = []
    for t, a, b in zip(texts, p2, p1):
        if b >= BLOCK_V1 and a >= BLOCK_V2:
            decision = "block"
        elif a >= review_threshold:
            decision = "review"
        else:
            decision = "ok"
        out.append({
            "text": t,
            "prob_abusive": round(float(a), 4),
            "prob_v1": round(float(b), 4),
            "decision": decision,
        })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="تشخیص کامنت نامناسب فارسی (ok/review/block)")
    ap.add_argument("texts", nargs="+", help="متن کامنت(ها)")
    args = ap.parse_args()
    icons = {"ok": "✅ ok", "review": "🔍 بازبینی", "block": "🚫 بلاک"}
    for r in predict(args.texts):
        print(f"{icons[r['decision']]:<12s} (p={r['prob_abusive']:.2f})  {r['text'][:80]}")
