# -*- coding: utf-8 -*-
"""تولید گزارش کامل مدل و داده‌ها → moderation/report.json

هر بار بعد از آموزش مجدد هم اجرا می‌شود (train_web.py صدا می‌زند) تا گزارش
همیشه با مدل فعال همگام باشد. صفحه /report/ سایت این فایل را می‌خواند.

محتوا: آمار داده (منبع‌به‌منبع، طول متن، برچسب)، نتایج کامل ۵ تست با ماتریس
درهم‌ریختگی و CI، جدول جابه‌جایی آستانه، آمار ویژگی‌ها و ضرایب مدل.
"""
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import normalize  # noqa: E402

MERGED = HERE.parent / 'data-merged'
TESTS = ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']
FA = {'tweets': 'توییتر', 'naseza': 'تلگرام (ناسزا)', 'pars_offensive': 'اینستاگرام (ParsOffensive)',
      'phate': 'توییتر (PHate)', 'phicad': 'اینستاگرام (PHICAD)'}
CLEANING = [
    {'step': 'دیتاست پایه (HuggingFace)', 'n': 33338,
     'note': 'persian-abusive-words — توییت با برچسب ۰/۱ (Apache-2.0)'},
    {'step': 'حذف تکراری‌ها پس از نرمال‌سازی', 'n': -2021, 'note': 'ادغام train/test رسمی و یکتاسازی'},
    {'step': 'حذف برچسب متناقض', 'n': -3, 'note': 'متن یکسان با دو برچسب مختلف'},
    {'step': 'دیتاست پایه پاک‌شده', 'n': 31314, 'note': 'تقسیم تمیز ۸۵/۱۵ با صفر هم‌پوشانی'},
    {'step': 'افزودن ۴ منبع (ناسزا، ParsOffensive، PHate، PHICAD)', 'n': None,
     'note': 'ناسزا CC0 — بقیه مصنوع تحقیقاتی'},
    {'step': 'مجموعه آموزش نهایی ادغام‌شده', 'n': 109747, 'note': '۴۵٫۷٪ مثبت — dedup بین‌منبعی'},
    {'step': 'گزینش ارزش‌محور (تحت ۲۰ هزار)', 'n': None,
     'note': 'آموزش متقاطع ۳-فولدی: نمونه‌های سخت (۸k) + مرزی (۵k) + پرکننده متنوع (۶٫۸k) — دسته در ستون reason'},
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
    lens = [len(t) for t, _ in rows]
    lens.sort()
    n = len(lens)
    pos = sum(l for _, l in rows)
    return {'n': n, 'pos': pos, 'neg': n - pos,
            'pos_pct': round(100 * pos / n, 1) if n else 0,
            'avg_len': round(sum(lens) / n, 1) if n else 0,
            'med_len': lens[n // 2] if n else 0,
            'p90_len': lens[int(n * 0.9)] if n else 0}


def boot_ci(y, pred, n_boot=800, seed=42):
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.RandomState(seed)
    from sklearn.metrics import f1_score
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
    from sklearn.metrics import precision_score, recall_score, f1_score
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {'P': round(P, 3), 'R': round(R, 3), 'F1': round(F, 3),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'n': tp + fp + fn + tn}


def generate():
    t0 = time.time()
    cfg = json.load(open(HERE / 'model_config.json', encoding='utf-8'))
    rt, bt1, bt2 = cfg['review_threshold'], cfg['block_v1'], cfg['block_v2']

    # ── آمار داده آموزش (به تفکیک منبع) ──
    sel_path = MERGED / 'train_selected.csv'
    per_src, all_rows = {}, []
    reasons = Counter()
    train_file = sel_path if sel_path.exists() else MERGED / 'train_merged.csv'
    with open(train_file, encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        for fl in r:
            if len(fl) >= 3:
                per_src.setdefault(fl[2], []).append((fl[0], int(fl[1])))
                if len(fl) >= 4:
                    reasons[fl[3]] += 1
    train_stats = {'total': 0, 'pos': 0, 'sources': []}
    for src, rows in sorted(per_src.items(), key=lambda kv: -len(kv[1])):
        b = stat_block(rows)
        b.update({'name': src, 'fa': FA.get(src, src)})
        train_stats['sources'].append(b)
        train_stats['total'] += b['n']; train_stats['pos'] += b['pos']
        all_rows += rows
    train_stats['pos_pct'] = round(100 * train_stats['pos'] / train_stats['total'], 1)

    # ── مدل‌ها و پیش‌بینی روی تست‌ها ──
    m2 = joblib.load(HERE / 'model.joblib')
    m1 = joblib.load(HERE / 'model_v1.joblib')
    results, tests_meta = {}, []
    all_y, all_pred, all_dec = [], [], []
    for name in TESTS:
        p_ = MERGED / f'test_{name}.csv'
        if not p_.exists():
            continue
        rows = load_csv(p_)
        X = [t for t, _ in rows]; y = np.array([l for _, l in rows])
        p2 = m2.predict_proba(X)[:, 1]; p1 = m1.predict_proba(X)[:, 1]
        pred = (p2 >= rt).astype(int)
        block = (p1 >= bt1) & (p2 >= bt2)
        dec = np.where(block, 2, np.where(p2 >= rt, 1, 0))
        res = prf(y, pred)
        res['ci'] = boot_ci(y, pred)
        bres = prf(y, block.astype(int))
        res['block'] = {'P': bres['P'], 'R': bres['R'], 'n': int(block.sum())}
        res['decisions'] = {'ok': int((dec == 0).sum()), 'review': int((dec == 1).sum()),
                            'block': int((dec == 2).sum())}
        results[name] = res
        meta = stat_block(rows); meta.update({'name': name, 'fa': FA.get(name, name)})
        tests_meta.append(meta)
        all_y += list(y); all_pred += list(pred); all_dec += list(dec)

    mixed = prf(np.array(all_y), np.array(all_pred))
    mixed['ci'] = boot_ci(np.array(all_y), np.array(all_pred))

    # ── جدول جابه‌جایی آستانه (روی همه تست‌ها با هم) ──
    sweep = []
    p2_all, y_all = [], []
    for name in TESTS:
        p_ = MERGED / f'test_{name}.csv'
        if not p_.exists():
            continue
        rows = load_csv(p_)
        p2_all += list(m2.predict_proba([t for t, _ in rows])[:, 1])
        y_all += [l for _, l in rows]
    p2_all = np.array(p2_all); y_all = np.array(y_all)
    for thr in (0.30, 0.40, rt, 0.50, 0.60, 0.70):
        s = prf(y_all, (p2_all >= thr).astype(int))
        s['thr'] = round(thr, 3)
        sweep.append(s)

    # ── آمار ویژگی‌ها و ضرایب ──
    feats = m2.named_steps['feats']
    char_v = len(feats.transformer_list[0][1].named_steps['tfidf'].vocabulary_)
    word_v = len(feats.transformer_list[1][1].vocabulary_)
    coef = m2.named_steps['clf'].coef_
    feat_stats = {'char_vocab': char_v, 'word_vocab': word_v,
                  'total': char_v + word_v,
                  'coef_nonzero': int((coef != 0).sum()), 'coef_total': int(coef.size),
                  'coef_absmax': round(float(abs(coef).max()), 2),
                  'model_mb': round((HERE / 'model.joblib').stat().st_size / 1e6, 1),
                  'model_v1_mb': round((HERE / 'model_v1.joblib').stat().st_size / 1e6, 1)}

    decisions_total = {'ok': int(sum(r['decisions']['ok'] for r in results.values())),
                       'review': int(sum(r['decisions']['review'] for r in results.values())),
                       'block': int(sum(r['decisions']['block'] for r in results.values()))}

    report = {'generated_at': time.strftime('%Y-%m-%d %H:%M'),
              'version': cfg.get('version', '?'), 'algo': cfg.get('algo', 'LogisticRegression'),
              'thresholds': {'review': rt, 'block_v1': bt1, 'block_v2': bt2},
              'train': train_stats, 'tests': tests_meta, 'results': results,
              'mixed': mixed, 'sweep': sweep, 'features': feat_stats,
              'decisions_total': decisions_total, 'cleaning': CLEANING,
              'selection': dict(reasons) if reasons else None,
              'gen_seconds': round(time.time() - t0, 1)}
    json.dump(report, open(HERE / 'report.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    return report


if __name__ == '__main__':
    r = generate()
    print('report.json ساخته شد در', r['gen_seconds'], 'ثانیه —',
          r['train']['total'], 'نمونه آموزش،', len(r['results']), 'تست')
