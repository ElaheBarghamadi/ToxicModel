# -*- coding: utf-8 -*-
"""آموزش مدل تشخیص کامنت نامناسب فارسی روی داده‌ی پاکسازی‌شده.

ویژگی‌ها: char TF-IDF (2-5) روی متن + نسخه بدون فاصله  ∪  word TF-IDF (1-2)
مدل‌ها: LogisticRegression (با احتمال کالیبره) و LinearSVC
انتخاب: بهترین F1 کلاس توهین‌آمیز در اعتبارسنجی 3-fold
خروجی: model.joblib + metrics.json
"""
import csv, json, sys, time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import normalize, dual_form  # noqa: E402

CLEAN = HERE.parent / 'persian-abusive-words' / 'clean'


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        r = csv.reader(f)
        next(r)
        return [(fl[0], int(fl[1])) for fl in r if len(fl) == 2]


X_tr, y_tr = zip(*load(CLEAN / 'train_clean.csv'))
X_te, y_te = zip(*load(CLEAN / 'test_clean.csv'))
X_tr, X_te = list(X_tr), list(X_te)          # متن‌ها از قبل نرمال‌شده‌اند
y_tr, y_te = np.array(y_tr), np.array(y_te)
print(f'train={len(X_tr)} test={len(X_te)}')


def union_feats():
    char = Pipeline([
        ('dual', FunctionTransformer(dual_form)),
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                  min_df=2, max_features=200000, sublinear_tf=True)),
    ])
    word = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\S+',
                           ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return FeatureUnion([('char', char), ('word', word)])


def lr_pipe(C):
    return Pipeline([('feats', union_feats()),
                     ('clf', LogisticRegression(C=C, class_weight='balanced',
                                                max_iter=4000, random_state=42))])


def svc_pipe(C):
    return Pipeline([('feats', union_feats()),
                     ('clf', LinearSVC(C=C, class_weight='balanced',
                                       random_state=42))])


cv = StratifiedKFold(3, shuffle=True, random_state=42)

# ---------- 1) مقایسه نامزدها با CV ----------
candidates = [(f'LogReg C={C}', lr_pipe(C), C) for C in (1.0, 3.0, 10.0)] + \
             [(f'LinearSVC C={C}', svc_pipe(C), C) for C in (0.5, 2.0)]
results = []
for name, pipe, C in candidates:
    t0 = time.time()
    scores = []
    for tr_i, va_i in cv.split(X_tr, y_tr):
        m = clone(pipe).fit([X_tr[i] for i in tr_i], y_tr[tr_i])
        scores.append(f1_score(y_tr[va_i], m.predict([X_tr[i] for i in va_i]), pos_label=1))
    rec = {'name': name, 'cv_f1_abusive': float(np.mean(scores)),
           'cv_std': float(np.std(scores)), 'fit_seconds': round(time.time() - t0, 1)}
    results.append(rec)
    print(rec)

best = max(results, key=lambda r: r['cv_f1_abusive'])
best_C = best['name'].split('=')[1]
print('\nBEST:', best['name'], '→ cv f1 =', round(best['cv_f1_abusive'], 4))

# ---------- 2) مدل نهایی (SVC را کالیبره می‌کنیم تا احتمال بدهد) ----------
if 'SVC' in best['name']:
    final = Pipeline([('feats', union_feats()),
                      ('clf', CalibratedClassifierCV(
                          LinearSVC(C=float(best_C), class_weight='balanced'), cv=3))])
else:
    final = lr_pipe(float(best_C))

# ---------- 3) انتخاب آستانه روی خروجی out-of-fold آموزش ----------
t0 = time.time()
oof = cross_val_predict(clone(final), X_tr, y_tr, cv=cv, method='predict_proba')[:, 1]
prec, rec, thr = precision_recall_curve(y_tr, oof)
f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
t_f1 = float(thr[int(np.argmax(f1s[:-1]))])
idx_p97 = np.where(prec[:-1] >= 0.97)[0]
t_p97 = float(thr[idx_p97[0]]) if len(idx_p97) else None
print(f'oof in {time.time()-t0:.0f}s | thresholds: default=0.5  f1-max={t_f1:.3f}  p>=0.97={t_p97 and round(t_p97,3)}')

# ---------- 4) آموزش نهایی و ارزیابی روی تست ----------
t0 = time.time()
final.fit(X_tr, y_tr)
print(f'final fit in {time.time()-t0:.0f}s')
p_te = final.predict_proba(X_te)[:, 1]

thresholds = {'default': 0.5, 'balanced_f1': t_f1}
if t_p97 is not None:
    thresholds['high_precision_0.97'] = t_p97

test_eval = {}
for tname, t in thresholds.items():
    pred = (p_te >= t).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum()); fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum()); tn = int(((pred == 0) & (y_te == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    test_eval[tname] = {
        'threshold': round(float(t), 4),
        'abusive_precision': round(P, 4), 'abusive_recall': round(R, 4),
        'abusive_f1': round(2 * P * R / max(P + R, 1e-9), 4),
        'accuracy': round(float((pred == y_te).mean()), 4),
        'confusion_TP_FP_FN_TN': [tp, fp, fn, tn],
    }
    print(f"\n--- test @{tname} (t={t:.3f}) ---")
    print(classification_report(y_te, pred, target_names=['normal(0)', 'abusive(1)'], digits=3))

# ---------- 5) آزمون روی کامنت‌های واقع‌گرایانه سایت ----------
civil = [
    'عالی بود ممنون از مدیریت سایت',
    'نویسنده محترم، با آمار این مقاله مخالفم و دلایل علمی‌اش را نوشتم',
    'ارسال کالا سه روز تأخیر داشت و پشتیبانی جواب نداد، خیلی ناراضی‌ام',
    'قیمت مناسبه ولی کیفیت دوام نداره',
    'چطور می‌تونم سفارشم رو لغو کنم؟',
    'به نظرم این سیاست جدید اثربخش نخواهد بود',
]
abusive = [
    'مدیریت این سایت یه عرفه',
    'نویسنده‌ش خودش یه الاغ بی‌سواده',
    'همه کارمندای این سایت دزد و کلاهبردارن',
    'این محصول گه بود، نخرید',
]
tricky = [
    'ک ی ر تو این سایت خراب شده',        # حروف جدا شده
    'بریمmmmm فاک این وضعیت',              # کشیدگی
]
demo = []
for txts, tag in [(civil, 'civil'), (abusive, 'abusive'), (tricky, 'tricky')]:
    for t in txts:
        p = float(final.predict_proba([normalize(t)])[0, 1])
        demo.append({'group': tag, 'text': t, 'prob_abusive': round(p, 3),
                     'flagged_at_0.5': bool(p >= 0.5)})
        print(f"[{tag:7s}] p={p:.3f}  {'FLAG' if p>=0.5 else 'ok  '}  {t[:70]}")

# سرعت استنتاج
t0 = time.time()
_ = final.predict_proba([normalize('یک کامنت تستی سرعت') ] * 1000)
latency = round((time.time() - t0), 2)
print(f'\ninference: 1000 comments in {latency}s')

# ---------- 6) ذخیره ----------
joblib.dump(final, HERE / 'model.joblib')
metrics = {'best_model': best['name'], 'cv_results': results,
           'thresholds': thresholds, 'test_eval': test_eval,
           'demo': demo, 'inference_1000_comments_seconds': latency,
           'data': {'train': len(X_tr), 'test': len(X_te),
                    'train_abusive': int(y_tr.sum()), 'test_abusive': int(y_te.sum())}}
with open(HERE / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print('\nsaved: model.joblib + metrics.json')
