# -*- coding: utf-8 -*-
"""نسخه ۴ — شروع تازه: فقط داده ایرانی، تک‌مدل LogisticRegression، بدون صف بازبینی.

تصمیم فقط دوحالته است:
    p ≥ block_threshold → 🚫 block (نامناسب)
    در غیر این صورت     → ✅ ok (مجاز)

مراحل: انتخاب C و آستانه روی validation جداگانه → آموزش نهایی روی کل داده انتخابی →
ارزیابی یک‌باره روی ۵ تست منبع + مجموعه تولیدی کامنت سایت.
"""
import csv, json, sys, time
from pathlib import Path
import joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import dual_form

MERGED = HERE.parent / 'data-merged'
TESTS = ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']
SEED = 42


def load(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        r = csv.reader(f); next(r)
        for fl in r:
            if len(fl) >= 2 and fl[1].strip() in ('0', '1'):
                rows.append((fl[0], int(fl[1])))
    return rows


def feats():
    char = Pipeline([('dual', FunctionTransformer(dual_form)),
                     ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                               min_df=3, max_features=200000,
                                               sublinear_tf=True, dtype=np.float32))])
    word = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\S+',
                           ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([('char', char), ('word', word)])


def lr(C):
    return Pipeline([('feats', feats()),
                     ('clf', LogisticRegression(C=C, class_weight='balanced',
                                                max_iter=1000, random_state=SEED))])


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {'P': round(P, 3), 'R': round(R, 3), 'F1': round(2 * P * R / max(P + R, 1e-9), 3),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


t0 = time.time()
full = load(MERGED / 'train_selected.csv')
tests = {n: load(MERGED / f'test_{n}.csv') for n in TESTS if (MERGED / f'test_{n}.csv').exists()}
website = load(HERE.parent / 'eval' / 'website_comments_test.csv')
print(f'train={len(full)} | tests={ {k: len(v) for k, v in tests.items()} } | website={len(website)}', flush=True)

tr, val = train_test_split(full, test_size=3500, stratify=[l for _, l in full], random_state=SEED)
Xtr, ytr = [t for t, _ in tr], np.array([l for _, l in tr])
Xv, yv = [t for t, _ in val], np.array([l for _, l in val])

# ---------- ۱) انتخاب C و آستانه روی validation ----------
best = None
for C in (1.0, 4.0, 10.0):
    m = lr(C).fit(Xtr, ytr)
    pv = m.predict_proba(Xv)[:, 1]
    prec, rec, thr = precision_recall_curve(yv, pv)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.argmax(f1s[:-1]))
    print(f'C={C}: val F1={f1s[i]:.4f} @ thr={thr[i]:.3f} (P={prec[i]:.3f} R={rec[i]:.3f})', flush=True)
    if best is None or f1s[i] > best[0]:
        best = (f1s[i], C, float(thr[i]))
F1_VAL, BEST_C, THR = best
print(f'⇒ C={BEST_C} | block_threshold={THR:.3f} (از validation)', flush=True)

# ---------- ۲) آموزش نهایی روی کل داده ----------
model = lr(BEST_C).fit([t for t, _ in full], np.array([l for _, l in full]))
print(f'final fit done ({time.time()-t0:.0f}s)', flush=True)

# ---------- ۳) ارزیابی یک‌باره ----------
report = {'per_test': {}}
print(f"\n{'test':>16s} |   P      R      F1")
for n, pairs in {**tests, 'website': website}.items():
    X = [t for t, _ in pairs]; y = np.array([l for _, l in pairs])
    pred = (model.predict_proba(X)[:, 1] >= THR).astype(int)
    r = prf(y, pred)
    report['per_test'][n] = r
    print(f'{n:>16s} | {r["P"]:.3f}  {r["R"]:.3f}  {r["F1"]:.3f}')

# ---------- ۴) ذخیره ----------
joblib.dump(model, HERE / 'model.joblib')
cfg = {'version': 'v4-ir', 'algo': f'LogisticRegression (C={BEST_C})',
       'block_threshold': round(THR, 4), 'selected_on': 'validation 3.5k',
       'val_f1': round(F1_VAL, 4), 'train_size': len(full),
       'trained_at': time.strftime('%Y-%m-%d %H:%M'), 'decisions': ['ok', 'block']}
json.dump(cfg, open(HERE / 'model_config.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
m = json.load(open(HERE / 'metrics.json', encoding='utf-8'))
m['v4'] = {**cfg, 'per_test': report['per_test']}
json.dump(m, open(HERE / 'metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'\ndone {time.time()-t0:.0f}s — model.joblib (v4-ir, تک‌مدل، بدون بازبینی)')
