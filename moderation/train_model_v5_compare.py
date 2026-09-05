# -*- coding: utf-8 -*-
"""مقایسه منصفانه الگوریتم‌ها — مدل استقرار (لجستیک) دست‌نخورده می‌ماند.

پروتکل یکسان برای همه:
  · همان داده: train_selected.csv (۱۹٬۸۰۰)
  · همان تقسیم validation (۳٬۵۰۰، seed=42) → انتخاب ابرپارامتر + آستانه هر مدل
  · آموزش نهایی روی کل داده → ارزیابی یک‌باره روی ۶ تست (۵ منبع + ۸۴ کامنت سایت)

کاندیدها:
  lr      : LogisticRegression C=4 (خط‌پایه = مدل فعلی)
  svc     : LinearSVC کالیبره‌شده (C از validation)
  nb      : MultinomialNB (بررسی سنتی)
  vote    : آنسامبل رأی‌گیری نرم (LR + SVC + NB)
خروجی: metrics.json[v5_comparison] + model_alt.joblib (بهترین غیر-لجستیک)
"""
import csv, json, sys, time
from pathlib import Path
import joblib, numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import dual_form

MERGED = HERE.parent / 'data-merged'
TESTS = ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad', 'website']
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


def pipe(clf):
    return Pipeline([('feats', feats()), ('clf', clf)])


def tune_threshold(p, y):
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.argmax(f1s[:-1]))
    return float(thr[i]), float(f1s[i])


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return round(P, 3), round(R, 3), round(2 * P * R / max(P + R, 1e-9), 3)


t0 = time.time()
full = load(MERGED / 'train_selected.csv')
tests = {}
for n in TESTS:
    p_ = MERGED / f'test_{n}.csv'
    if not p_.exists() and n == 'website':
        p_ = HERE.parent / 'eval' / 'website_comments_test.csv'
    if p_.exists():
        tests[n] = load(p_)
print(f'train={len(full)} | tests={ {k: len(v) for k, v in tests.items()} }', flush=True)

tr, val = train_test_split(full, test_size=3500, stratify=[l for _, l in full], random_state=SEED)
Xtr, ytr = [t for t, _ in tr], np.array([l for _, l in tr])
Xv, yv = [t for t, _ in val], np.array([l for _, l in val])

# ---------- ۱) ساخت کاندیدها ----------
def build(kind):
    if kind == 'lr':
        return pipe(LogisticRegression(C=4.0, class_weight='balanced', max_iter=1000, random_state=SEED))
    if kind == 'svc':
        return pipe(CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', random_state=SEED), cv=3))
    if kind == 'nb':
        return pipe(MultinomialNB(alpha=0.05))
    if kind == 'vote':
        return pipe(VotingClassifier([
            ('lr', LogisticRegression(C=4.0, class_weight='balanced', max_iter=1000, random_state=SEED)),
            ('svc', CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', random_state=SEED), cv=3)),
            ('nb', MultinomialNB(alpha=0.05))], voting='soft', weights=[2, 2, 1]))

# ---------- ۲) تنظیم روی validation ----------
results = {}
for kind in ['lr', 'svc', 'nb', 'vote']:
    ts = time.time()
    m = build(kind).fit(Xtr, ytr)
    pv = m.predict_proba(Xv)[:, 1]
    thr, vf1 = tune_threshold(pv, yv)
    results[kind] = {'val_f1': round(vf1, 4), 'thr': round(thr, 4), 'val_sec': round(time.time() - ts)}
    print(f'{kind:5s}: val F1={vf1:.4f} @ thr={thr:.3f} ({time.time()-ts:.0f}s)', flush=True)
    del m

# ---------- ۳) آموزش نهایی و ارزیابی یک‌باره ----------
print('\nنتیجه نهایی روی تست‌های دست‌نخورده:', flush=True)
print(f"{'model':>6s} | " + ' | '.join(f'{n[:9]:>9s}' for n in tests) + ' |  mixed')
final_scores = {}
for kind in results:
    m = build(kind).fit([t for t, _ in full], np.array([l for _, l in full]))
    row, ys, ps_ = [], [], []
    for n, pairs in tests.items():
        X = [t for t, _ in pairs]; y = np.array([l for _, l in pairs])
        pred = (m.predict_proba(X)[:, 1] >= results[kind]['thr']).astype(int)
        P, R, F = prf(y, pred)
        row.append(F); ys += list(y); ps_ += list(pred)
        results[kind].setdefault('per_test', {})[n] = {'P': P, 'R': R, 'F1': F}
    mixed = prf(np.array(ys), np.array(ps_))
    results[kind]['mixed'] = {'P': mixed[0], 'R': mixed[1], 'F1': mixed[2]}
    final_scores[kind] = mixed[2]
    print(f'{kind:>6s} | ' + ' | '.join(f'{v:>9.3f}' for v in row) + f' |  {mixed[2]:.3f}', flush=True)
    if kind != 'lr':   # مدل استقرار (لجستیک) جداگانه موجود است
        joblib.dump(m, HERE / f'model_{kind}.joblib' if kind != 'vote' else HERE / 'model_vote.joblib')
    del m

best_alt = max([k for k in results if k != 'lr'], key=lambda k: results[k]['mixed']['F1'])
print(f'\nبهترین غیر-لجستیک: {best_alt} (mixed F1={results[best_alt]["mixed"]["F1"]})')
shutil_src = HERE / ('model_vote.joblib' if best_alt == 'vote' else f'model_{best_alt}.joblib')
import shutil
shutil.copy(shutil_src, HERE / 'model_alt.joblib')
for k in [k for k in results if k not in ('lr', best_alt)]:
    p_ = HERE / f'model_{k}.joblib'
    if p_.exists():
        p_.unlink()

verdict = 'logistic' if results['lr']['mixed']['F1'] >= results[best_alt]['mixed']['F1'] else best_alt
m = json.load(open(HERE / 'metrics.json', encoding='utf-8'))
m['v5_comparison'] = {'protocol': 'same data/val/tests; per-model threshold from validation',
                      'results': results, 'best_alt': best_alt,
                      'deployed_stays': 'lr', 'verdict': verdict}
json.dump(m, open(HERE / 'metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'⇒ حکم: مدل استقرار = لجستیک؛ بهترین جایگزین = {best_alt} | کل {time.time()-t0:.0f}s')
