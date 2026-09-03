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
    """پاک‌کردن کش مدل‌ها (بعد از آموزش مجدد/بازگردانی)."""
    moderate.reload_models()


def model_config():
    """پیکربندی فعلی مدل (آستانه‌ها، نسخه، الگوریتم)."""
    path = MOD / 'model_config.json'
    try:
        return json.load(open(path, encoding='utf-8'))
    except Exception:
        return {'version': 'نامشخص', 'algo': 'LogisticRegression',
                'review_threshold': 0.448, 'block_v1': 0.8, 'block_v2': 0.9}


def archived_models():
    """نسخه‌های بایگانی‌شده (جدیدترین اول)."""
    arc = MOD / 'models_archive'
    if not arc.exists():
        return []
    return sorted(arc.glob('model_*.joblib'), reverse=True)


def rollback_model():
    """بازگردانی جدیدترین نسخه بایگانی به‌عنوان مدل فعال."""
    arcs = archived_models()
    if not arcs:
        return False, 'نسخه بایگانی‌ای وجود ندارد'
    import shutil
    shutil.copy(arcs[0], MOD / 'model.joblib')
    cfg = model_config()
    cfg['version'] = f'rollback←{arcs[0].stem}'
    cfg['trained_at'] = arcs[0].stem.replace('model_', '')
    json.dump(cfg, open(MOD / 'model_config.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    reload_models()
    return True, f'بازگردانی شد: {arcs[0].name}'


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
