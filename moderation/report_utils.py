# -*- coding: utf-8 -*-
"""تولید گزارش کامل مدل و داده‌ها → moderation/report.json (نسخه ۴ — تک‌مدل)."""
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MERGED = HERE.parent / 'data-merged'
TESTS = ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']
FA = {'tweets': 'توییتر', 'naseza': 'تلگرام (ناسزا)', 'pars_offensive': 'اینستاگرام (ParsOffensive)',
      'phate': 'توییتر (PHate)', 'phicad': 'اینستاگرام (PHICAD)', 'website': 'کامنت سایت (تولادی)'}
CLEANING = [
    {'step': 'شروع تازه (v4) — فقط منابع ایرانی', 'n': None,
     'note': '۵ منبع: توییتر×۲، تلگرام، اینستاگرام×۲ — بدون فارسی-دری'},
    {'step': 'دیتاست پایه (HuggingFace)', 'n': 33338, 'note': 'persian-abusive-words (Apache-2.0)'},
    {'step': 'پاکسازی: تکرار و تناقض برچسب', 'n': -2024, 'note': '→ ۳۱٬۳۱۴ یکتا؛ تقسیم تمیز ۸۵/۱۵'},
    {'step': 'ادغام ۵ منبع + dedup بین‌منبعی', 'n': 109747, 'note': 'صفر هم‌پوشانی آموزش/تست'},
    {'step': 'گزینش ارزش‌محور (زیر ۲۰ هزار)', 'n': 19800,
     'note': '۱k سخت + ۱k مرزی + ۱۷٫۸k طبقه‌بندی‌شده (آزمایش ترکیب‌ها روی validation)'},
]


def load_csv(path):
    out = []
    with open(path, encoding='utf-8-sig') as f:
        r = csv.reader(f)
        next(r, None)
        for fl in r:
            if len(fl) >= 2 and fl[1].strip() in ('0', '1'):
                out.append((fl[0], int(fl[1])))
    return out


def stat_block(rows):
    lens = sorted(len(t) for t, _ in rows)
    n = len(lens)
    pos = sum(l for _, l in rows)
    return {'n': n, 'pos': pos, 'neg': n - pos,
            'pos_pct': round(100 * pos / n, 1) if n else 0,
            'avg_len': round(sum(lens) / n, 1) if n else 0,
            'med_len': lens[n // 2] if n else 0,
            'p90_len': lens[int(n * 0.9)] if n else 0}


def boot_ci(y, pred, n_boot=800, seed=42):
    from sklearn.metrics import f1_score
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(f1_score(y[i], pred[i], pos_label=1, zero_division=0))
    if not vals:
        return [0.0, 0.0]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {'P': round(P, 3), 'R': round(R, 3), 'F1': round(2 * P * R / max(P + R, 1e-9), 3),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n': tp + fp + fn + tn}


def generate():
    t0 = time.time()
    cfg = json.load(open(HERE / 'model_config.json', encoding='utf-8'))
    thr = cfg.get('block_threshold', 0.509)

    per_src = {}
    sel_path = MERGED / 'train_selected.csv'
    train_file = sel_path if sel_path.exists() else MERGED / 'train_merged.csv'
    reasons = Counter()
    with open(train_file, encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        for fl in r:
            if len(fl) >= 3:
                per_src.setdefault(fl[2], []).append((fl[0], int(fl[1])))
                if len(fl) >= 4:
                    reasons[fl[3]] += 1
    train_stats = {'total': 0, 'pos': 0, 'sources': []}
    for src, rows in sorted(per_src.items(), key=lambda kv: -len(kv[1])):
        b = stat_block(rows); b.update({'name': src, 'fa': FA.get(src, src)})
        train_stats['sources'].append(b)
        train_stats['total'] += b['n']; train_stats['pos'] += b['pos']
    train_stats['pos_pct'] = round(100 * train_stats['pos'] / max(train_stats['total'], 1), 1)

    model = joblib.load(HERE / 'model.joblib')
    results, tests_meta = [], []
    all_y, all_pred, all_dec = [], [], []
    p2_all = []
    for name in TESTS + ['website']:
        p_ = MERGED / f'test_{name}.csv'
        if not p_.exists():
            wp = HERE.parent / 'eval' / 'website_comments_test.csv'
            if name == 'website' and wp.exists():
                p_ = wp
            else:
                continue
        rows = load_csv(p_)
        X = [t for t, _ in rows]; y = np.array([l for _, l in rows])
        probs = model.predict_proba(X)[:, 1]
        pred = (probs >= thr).astype(int)
        res = prf(y, pred)
        res['ci'] = boot_ci(y, pred)
        res['decisions'] = {'ok': int((pred == 0).sum()), 'block': int((pred == 1).sum())}
        results.append((name, res))
        meta = stat_block(rows); meta.update({'name': name, 'fa': FA.get(name, name)})
        tests_meta.append(meta)
        all_y += list(y); all_pred += list(pred)
        p2_all += list(probs)
    results = dict(results)
    mixed = prf(np.array(all_y), np.array(all_pred))
    mixed['ci'] = boot_ci(np.array(all_y), np.array(all_pred))

    sweep = []
    p2_all = np.array(p2_all); y_all = np.array(all_y)
    for t in (0.30, 0.40, thr, 0.50, 0.60, 0.70):
        s = prf(y_all, (p2_all >= t).astype(int))
        s['thr'] = round(t, 3)
        sweep.append(s)

    feats = model.named_steps['feats']
    char_v = len(feats.transformer_list[0][1].named_steps['tfidf'].vocabulary_)
    word_v = len(feats.transformer_list[1][1].vocabulary_)
    coef = model.named_steps['clf'].coef_
    feat_stats = {'char_vocab': char_v, 'word_vocab': word_v, 'total': char_v + word_v,
                  'coef_nonzero': int((coef != 0).sum()), 'coef_total': int(coef.size),
                  'model_mb': round((HERE / 'model.joblib').stat().st_size / 1e6, 1)}

    dec_total = {'ok': int(sum(r['decisions']['ok'] for r in results.values())),
                 'block': int(sum(r['decisions']['block'] for r in results.values()))}
    report = {'generated_at': time.strftime('%Y-%m-%d %H:%M'),
              'version': cfg.get('version', '?'), 'algo': cfg.get('algo', 'LogisticRegression'),
              'thresholds': {'block': thr},
              'train': train_stats, 'tests': tests_meta, 'results': results,
              'mixed': mixed, 'sweep': sweep, 'features': feat_stats,
              'decisions_total': dec_total, 'cleaning': CLEANING,
              'selection': dict(reasons) if reasons else None,
              'gen_seconds': round(time.time() - t0, 1)}
    json.dump(report, open(HERE / 'report.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    return report


if __name__ == '__main__':
    r = generate()
    print('report.json:', r['train']['total'], 'نمونه |', len(r['results']), 'تست | mixed F1 =',
          r['mixed']['F1'], r['mixed']['ci'])
