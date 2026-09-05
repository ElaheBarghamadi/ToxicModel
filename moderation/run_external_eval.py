# -*- coding: utf-8 -*-
"""ارزیابی مدل فعال روی دو مجموعه تست مستقل:

  A) eval/website_comments_test.csv — ۸۴ کامنت تولیدی واقع‌گرایانه (سایت فارسی)
  B) data-merged/test_farsi_dari.csv — ۲٬۴۹۹ کامنت واقعی فارسی-دری (کگل، CC0، دیده‌نشده توسط مدل)

«پیش‌بینی مثبت» = تصمیم review یا block (آستانه‌های فعال از model_config.json)
"""
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moderate import predict  # noqa: E402

ROOT = HERE.parent


def load(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        r = csv.reader(f)
        next(r)
        for fl in r:
            if len(fl) >= 2 and fl[1].strip() in ('0', '1'):
                rows.append((fl[0], int(fl[1]), fl[2] if len(fl) > 2 else ''))
    return rows


def score(rows, title):
    res = predict([t for t, _, _ in rows])
    y = np.array([l for _, l, _ in rows])
    pred = np.array([1 if r['decision'] in ('review', 'block') else 0 for r in res])
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    acc = (tp + tn) / len(rows)
    dec = Counter(r['decision'] for r in res)
    print(f'\n════ {title} ════')
    print(f'  n={len(rows)} | دقت={P:.3f} بازیابی={R:.3f} F1={F1:.3f} Accuracy={acc:.3f}')
    print(f'  ماتریس: TP={tp} FP={fp} FN={fn} TN={tn} | تصمیم‌ها: {dict(dec)}')
    return res, y, pred


def main():
    cfg = json.load(open(HERE / 'model_config.json', encoding='utf-8'))
    print(f'مدل فعال: {cfg.get("version")} | آستانه‌ها: review≥{cfg["review_threshold"]}, '
          f'block=(v1≥{cfg["block_v1"]} ∧ v2≥{cfg["block_v2"]})')

    # ── A) کامنت‌های تولیدی سایت ──
    gen = load(ROOT / 'eval' / 'website_comments_test.csv')
    t0 = time.time()
    res_g, y_g, pred_g = score(gen, 'A) کامنت‌های تولیدی سایت فارسی (۸۴ نمونه)')
    dt = time.time() - t0

    per_cat = defaultdict(lambda: [0, 0])
    errors = []
    for (t, l, cat), r, p in zip(gen, res_g, pred_g):
        per_cat[cat][1] += 1
        if p == l:
            per_cat[cat][0] += 1
        else:
            errors.append((cat, l, r['decision'], round(r['prob_abusive'], 2), t))
    print('\n  دقت به تفکیک سناریو:')
    for cat, (ok, n) in sorted(per_cat.items()):
        print(f'    {cat:12s} {ok}/{n}' + ('  ⚠️' if ok < n else '  ✓'))
    print(f'\n  خطاها ({len(errors)}):')
    for cat, l, dec, p, t in errors:
        print(f'    [{cat}] واقعی={l} تصمیم={dec} p={p} | {t[:64]}')
    print(f'\n  سرعت: {len(gen)/dt:.0f} کامنت/ثانیه')

    # ── B) کامنت‌های واقعی دری (دیده‌نشده) ──
    dari = load(ROOT / 'data-merged' / 'test_farsi_dari.csv')
    score(dari, 'B) کامنت‌های واقعی فارسی-دری (۲٬۴۹۹ نمونه، خارج از دامنه)')


if __name__ == '__main__':
    main()
