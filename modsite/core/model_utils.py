# -*- coding: utf-8 -*-
"""پل بین جنگو و مدل‌های joblib (moderate.py)."""
import json
import sys
from pathlib import Path

from django.conf import settings

MOD = Path(settings.MODERATION_DIR)
if str(MOD) not in sys.path:
    sys.path.insert(0, str(MOD))

import moderate  # noqa: E402


def predict_rows(texts):
    """پیش‌بینی برای لیستی از متن‌ها — همان predict Moderate."""
    return moderate.predict(list(texts))


def reload_models():
    """پاک‌کردن کش مدل‌ها (بعد از آموزش مجدد)."""
    moderate._models.clear()


def load_metrics():
    """خواندن metrics.json با مقادیر پیش‌فرض امن."""
    path = MOD / 'metrics.json'
    try:
        return json.load(open(path, encoding='utf-8'))
    except Exception:
        return {}


def train_data_stats():
    """آمار مجموعه آموزش ادغام‌شده (کش‌شده)."""
    path = Path(settings.DATA_MERGED_DIR) / 'train_merged.csv'
    if not path.exists():
        return {'total': 0, 'sources': {}, 'pos_ratio': None}
    import csv
    from collections import Counter
    src, labs = Counter(), Counter()
    with open(path, encoding='utf-8') as f:
        r = csv.reader(f)
        next(r)
        for fl in r:
            if len(fl) >= 3:
                src[fl[2]] += 1
                labs[fl[1]] += 1
    total = sum(src.values())
    return {'total': total, 'sources': dict(src),
            'pos_ratio': round(labs.get('1', 0) / total, 3) if total else None}
