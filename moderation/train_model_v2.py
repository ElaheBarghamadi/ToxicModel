# -*- coding: utf-8 -*-
"""آموزش مدل v2 روی دیتاست ادغام‌شده ۵ منبعی (توییتر+تلگرام+اینستاگرام×2+PHate).

میان‌برهای حافظه: dtype=float32، min_df=4، متن‌ها کوتاه (<=250 نویسه)
کاندیداها: LR C=1/4/16 (احتمال بومی) — انتخاب با CV روی زیرنمونه ۴۰هزار
خروجی: model.joblib (v2) + metrics.json + بکاپ v1
"""
import csv, json, shutil, sys, time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import dual_form  # noqa: E402

MERGED = HERE.parent / 'data-merged'


def load(path):
    with open(path, encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        return [(fl[0], int(fl[1])) for fl in r if len(fl) >= 2]


train = load(MERGED / 'train_merged.csv')
tests = {n: load(MERGED / f'test_{n}.csv') for n in
         ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']}
X = [t for t, _ in train]; y = np.array([l for _, l in train])
print(f'train={len(X)} pos={y.mean():.1%}', flush=True)


def union_feats():
    char = Pipeline([('dual', FunctionTransformer(dual_form)),
                     ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                               min_df=4, max_features=250000,
                                               sublinear_tf=True, dtype=np.float32))])
    word = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\S+',
                           ngram_range=(1, 2), min_df=4, sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([('char', char), ('word', word)])


def lr(C):
    return Pipeline([('feats', union_feats()),
                     ('clf', LogisticRegression(C=C, class_weight='balanced',
                                                max_iter=800, random_state=42))])


# ---------- 1) انتخاب C با CV روی زیرنمونه ----------
rng = np.random.RandomState(42)
idx = rng.choice(len(X), size=min(40000, len(X)), replace=False)
Xs, ys = [X[i] for i in idx], y[idx]
skf = StratifiedKFold(3, shuffle=True, random_state=42)
best_c, best_f1 = None, -1
for C in (1.0, 4.0, 16.0):
    t0 = time.time(); scores = []
    for tr, va in skf.split(Xs, ys):
        m = lr(C).fit([Xs[i] for i in tr], ys[tr])
        scores.append(f1_score(ys[va], m.predict([Xs[i] for i in va]), pos_label=1))
    print(f'LR C={C}: cv_f1={np.mean(scores):.4f} ±{np.std(scores):.4f} ({time.time()-t0:.0f}s)', flush=True)
    if np.mean(scores) > best_f1:
        best_c, best_f1 = C, np.mean(scores)
print('best C =', best_c, flush=True)

# ---------- 2) آموزش نهایی ----------
t0 = time.time()
model = lr(best_c).fit(X, y)
print(f'final fit {time.time()-t0:.0f}s', flush=True)

# ---------- 3) ارزیابی: مدل جدید vs مدل قدیمی(v1) روی همه تست‌ها ----------
v1 = joblib.load(HERE / 'model.joblib')


def prf(model, pairs, thr=0.5):
    xt = [t for t, _ in pairs]; yt = np.array([l for _, l in pairs])
    p = model.predict_proba(xt)[:, 1]
    pred = (p >= thr).astype(int)
    return (precision_score(yt, pred, pos_label=1, zero_division=0),
            recall_score(yt, pred, pos_label=1, zero_division=0),
            f1_score(yt, pred, pos_label=1, zero_division=0))


report = {'best_C': best_c, 'cv_f1_sample': round(best_f1, 4), 'train_size': len(X),
          'per_test': {}}
print(f"\n{'تست':>16s} | {'v1 P':>6s} {'v1 R':>6s} {'v1 F1':>6s} | {'v2 P':>6s} {'v2 R':>6s} {'v2 F1':>6s}")
for name, pairs in tests.items():
    a = prf(v1, pairs); b = prf(model, pairs)
    report['per_test'][name] = {
        'v1': dict(zip('P R F1'.split(), map(lambda v: round(v, 3), a))),
        'v2': dict(zip('P R F1'.split(), map(lambda v: round(v, 3), b))),
        'n': len(pairs), 'pos': int(sum(l for _, l in pairs))}
    print(f"{name:>16s} | {a[0]:.3f}  {a[1]:.3f}  {a[2]:.3f} | {b[0]:.3f}  {b[1]:.3f}  {b[2]:.3f}")

# ---------- 4) آزمون ضعف‌های شناخته‌شده v1 ----------
demo = {
    'civil': ['عالی بود ممنون از مدیریت سایت',
              'نویسنده محترم، با آمار این مقاله مخالفم',
              'ارسال کالا سه روز تأخیر داشت و پشتیبانی جواب نداد، ناراضی‌ام',
              'چطور می‌تونم سفارشم رو لغو کنم؟'],
    'mild_insult_v1_missed': ['نویسنده‌ش خودش یه الاغ بی‌سواده',
                              'همه کارمندای این سایت دزد و کلاهبردارن',
                              'مدیریت این سایت یه عرفه',
                              'بی غیرت هر کی این خبر رو نوشته',
                              'چی می‌گی دیوونه؟'],
    'profanity': ['بریم کیر تو این سایت خراب شده',
                  'ک ی ر تو شونصد تا تیکت باز کردم'],
}
report['demo'] = {}
for grp, texts in demo.items():
    report['demo'][grp] = []
    for t in texts:
        p1 = float(v1.predict_proba([t])[0, 1]); p2 = float(model.predict_proba([t])[0, 1])
        report['demo'][grp].append({'text': t, 'v1': round(p1, 3), 'v2': round(p2, 3)})
        print(f"[{grp[:5]}] v1={p1:.3f} → v2={p2:.3f}  {t[:60]}")

# ---------- 5) سرعت و ذخیره ----------
t0 = time.time()
_ = model.predict_proba(['کامنت تست'] * 2000)
report['inference_2000_sec'] = round(time.time() - t0, 3)
shutil.copy(HERE / 'model.joblib', HERE / 'model_v1.joblib')
joblib.dump(model, HERE / 'model.joblib')
m = json.load(open(HERE / 'metrics.json', encoding='utf-8'))
m['v2'] = report
json.dump(m, open(HERE / 'metrics.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nsaved: model.joblib (v2) | backup v1 | metrics.json updated', flush=True)
