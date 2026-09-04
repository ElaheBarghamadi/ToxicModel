# -*- coding: utf-8 -*-
"""نسخه ۳ — اصلاحات داوری + الزام «رگرسیون لجستیک»:

اصلاح ۱ (تطبیق‌یافته با الزام لجستیک):
    هر دو مدل (v1 و v2) LogisticRegression می‌شوند — LinearSVC حذف شد.
اصلاح ۲ (نشت ارزیابی آستانه‌ها):
    آستانه‌ی review و قاعده‌ی block فقط روی یک validation جداگانه انتخاب می‌شوند؛
    تست‌ها فقط یک‌بار و در پایان خوانده می‌شوند + فاصله اطمینان بوت‌استرپ ۹۵٪.
اصلاح ۳ (نسخه‌بندی):
    قبل از بازنویسی، از مدل فعلی بکاپ (model_prev) گرفته می‌شود + model_config.json.

خروجی: model.joblib (v2-LR) · model_v1.joblib (v1-LR) · model_config.json · metrics.json[v3]
"""
import csv, json, shutil, sys, time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import dual_form  # noqa: E402

MERGED = HERE.parent / 'data-merged'
TESTS = ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']
SEED = 42


def load(path):
    with open(path, encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        return [(fl[0], int(fl[1])) for fl in r if len(fl) >= 2]


def feats():
    char = Pipeline([('dual', FunctionTransformer(dual_form)),
                     ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                               min_df=4, max_features=250000,
                                               sublinear_tf=True, dtype=np.float32))])
    word = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\S+',
                           ngram_range=(1, 2), min_df=4, sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([('char', char), ('word', word)])


def logreg(C):
    return Pipeline([('feats', feats()),
                     ('clf', LogisticRegression(C=C, class_weight='balanced',
                                                max_iter=1000, random_state=SEED))])


def boot_ci(y, pred, n=1000, seed=SEED):
    """فاصله اطمینان ۹۵٪ برای F1 با بوت‌استرپ."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y); pred = np.asarray(pred)
    f1s = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        f1s.append(f1_score(y[i], pred[i], pos_label=1, zero_division=0))
    return (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)))


t0 = time.time()
# ---------- بارگذاری (گزینش ارزش‌محور اگر موجود بود) ----------
sel_path = MERGED / 'train_selected.csv'
full_v2 = load(sel_path) if sel_path.exists() else load(MERGED / 'train_merged.csv')
if sel_path.exists():
    with open(sel_path, encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        full_v1 = [(fl[0], int(fl[1])) for fl in r
                   if len(fl) >= 3 and fl[2] == 'tweets' and fl[1].strip() in ('0', '1')]
else:
    full_v1 = load(HERE.parent / 'persian-abusive-words' / 'clean' / 'train_clean.csv')
tests = {n: load(MERGED / f'test_{n}.csv') for n in TESTS if (MERGED / f'test_{n}.csv').exists()}
print(f'v2-train={len(full_v2)} (گزینش‌شده: {sel_path.exists()}) | v1-train={len(full_v1)} | tests={ {k: len(v) for k,v in tests.items()} }', flush=True)

# ----------validation جداگانه (اصلاح ۲) ----------
v2_tr, v2_val = train_test_split(full_v2, test_size=min(4000, max(1500, int(len(full_v2)*0.2))),
                                 stratify=[l for _, l in full_v2], random_state=SEED)
v1_tr, v1_val = train_test_split(full_v1, test_size=min(1500, max(600, int(len(full_v1)*0.2))),
                                 stratify=[l for _, l in full_v1], random_state=SEED)
print(f'validation: v2={len(v2_val)} | v1={len(v1_val)} (آستانه‌ها فقط این‌جا تنظیم می‌شوند)', flush=True)

# ---------- آموزش مدل‌های آزمایشی روی train-minus-val ----------
m1 = logreg(10.0).fit([t for t, _ in v1_tr], [l for _, l in v1_tr])
m2 = logreg(4.0).fit([t for t, _ in v2_tr], [l for _, l in v2_tr])
print(f'temp fits done ({time.time()-t0:.0f}s)', flush=True)

Xv = [t for t, _ in v2_val]; yv = np.array([l for _, l in v2_val])
p1v = m1.predict_proba(Xv)[:, 1]
p2v = m2.predict_proba(Xv)[:, 1]

# آستانه review: بیشینه F1 روی validation
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(yv, p2v)
f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
REVIEW_T = float(thr[int(np.argmax(f1s[:-1]))])
REVIEW_T = min(max(REVIEW_T, 0.3), 0.7)  # مهار در بازه معقول
print(f'REVIEW threshold (از validation) = {REVIEW_T:.3f}', flush=True)

# قاعده block: grid روی validation — هدف precision ≥ 0.97؛ اگر نشد، بهترین precision موجود
best = None
fallback = None
for t1 in (0.80, 0.85, 0.90, 0.95, 0.97):
    for t2 in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        b = (p1v >= t1) & (p2v >= t2)
        nb = int(b.sum())
        if nb < 25:
            continue
        Pb = precision_score(yv, b, zero_division=0)
        Rb = recall_score(yv, b)
        if fallback is None or Pb > fallback[3] or (Pb == fallback[3] and Rb > fallback[0]):
            fallback = (Rb, t1, t2, Pb, nb)
        if Pb >= 0.97 and (best is None or Rb > best[0]):
            best = (Rb, t1, t2, Pb, nb)
chosen = best or fallback
BLOCK_V1, BLOCK_V2 = chosen[1], chosen[2]
mode = 'P≥0.97' if best else f'best-available P={chosen[3]:.3f}'
print(f'BLOCK rule (از validation): v1>={BLOCK_V1} و v2>={BLOCK_V2} | {mode} | R={chosen[0]:.3f} n={chosen[4]}', flush=True)

# ---------- آموزش نهایی روی کل داده آموزش ----------
m1_final = logreg(10.0).fit([t for t, _ in full_v1], [l for _, l in full_v1])
m2_final = logreg(4.0).fit([t for t, _ in full_v2], [l for _, l in full_v2])
print(f'final fits done ({time.time()-t0:.0f}s)', flush=True)

# ---------- ارزیابی یک‌باره روی تست‌های دست‌نخورده + CI ----------
report = {'review_threshold': round(REVIEW_T, 4), 'block_v1': BLOCK_V1, 'block_v2': BLOCK_V2,
          'selected_on': 'validation (8k from train)', 'algo': 'LogisticRegression (both models)',
          'per_test': {}}
print(f"\n{'test':>16s} | {'P':>6s} {'R':>6s} {'F1':>6s} {'CI95':>15s} | block P/R")
for n, pairs in tests.items():
    Xt = [t for t, _ in pairs]; yt = np.array([l for _, l in pairs])
    p2 = m2_final.predict_proba(Xt)[:, 1]
    p1 = m1_final.predict_proba(Xt)[:, 1]
    pred = (p2 >= REVIEW_T).astype(int)
    P = precision_score(yt, pred, pos_label=1, zero_division=0)
    R = recall_score(yt, pred, pos_label=1, zero_division=0)
    F = f1_score(yt, pred, pos_label=1, zero_division=0)
    lo, hi = boot_ci(yt, pred)
    b = (p1 >= BLOCK_V1) & (p2 >= BLOCK_V2)
    Pb = precision_score(yt, b, zero_division=0); Rb = recall_score(yt, b, zero_division=0)
    report['per_test'][n] = {'P': round(P, 3), 'R': round(R, 3), 'F1': round(F, 3),
                             'F1_ci95': [round(lo, 3), round(hi, 3)],
                             'block_P': round(Pb, 3), 'block_R': round(Rb, 3), 'n': len(pairs)}
    print(f'{n:>16s} | {P:.3f} {R:.3f} {F:.3f} [{lo:.3f},{hi:.3f}] | {Pb:.3f}/{Rb:.3f}')

# ---------- ذخیره با نسخه‌بندی (اصلاح ۳) ----------
for cur, bak in [('model.joblib', 'model_prev.joblib'), ('model_v1.joblib', 'model_v1_prev.joblib')]:
    if (HERE / cur).exists():
        shutil.copy(HERE / cur, HERE / bak)
joblib.dump(m2_final, HERE / 'model.joblib')
joblib.dump(m1_final, HERE / 'model_v1.joblib')
cfg = {**report, 'version': 'v3', 'trained_at': time.strftime('%Y-%m-%d %H:%M'),
       'train_size_v2': len(full_v2), 'train_size_v1': len(full_v1)}
json.dump(cfg, open(HERE / 'model_config.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

m = json.load(open(HERE / 'metrics.json', encoding='utf-8'))
m['v3'] = cfg
json.dump(m, open(HERE / 'metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'\ndone in {time.time()-t0:.0f}s — model.joblib(v2-LR) + model_v1.joblib(v1-LR) + model_config.json', flush=True)
