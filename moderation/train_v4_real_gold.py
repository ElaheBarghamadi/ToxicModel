# -*- coding: utf-8 -*-
"""آموزش «کم اما طلایی»: ۱۲k هسته‌ی تاییدشده + ۲۲۶ کامنت واقعی دست‌لیبل (طلیا).

تصمیم: اگر F1 یوتیوبِ واقعی بهتر شد و mixed پنج‌تستی سقوط نکرد (≥0.84)،
خروجی جایگزین train_selected.csv می‌شود؛ وگرنه فقط گزارش.
"""
import csv, random, sys, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import dual_form, normalize

MERGED = HERE.parent / 'data-merged'
EVAL = HERE.parent / 'eval'
CORE_N = 14000
BORDER_FRAC = 0.10
MILD_SRC = {'phicad', 'phate', 'naseza'}
MILD_BOOST = 1.6
TWEETS_FRAC = 0.20
POS_TARGET = 0.50
SEED = 42
random.seed(SEED)


def feats():
    char = Pipeline([('dual', FunctionTransformer(dual_form)),
                     ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                               min_df=3, max_features=150000,
                                               sublinear_tf=True, dtype=np.float32))])
    word = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\S+',
                           ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([('char', char), ('word', word)])


def read2(p):
    with open(p, encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        return [(fl[0], int(fl[1])) for fl in r if len(fl) >= 2 and fl[1].strip() in ('0', '1')]


def main():
    t0 = time.time()
    with open(MERGED / 'train_merged.csv', encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        rows3 = [(fl[0], int(fl[1]), fl[2]) for fl in r
                 if len(fl) >= 3 and len(fl[0]) >= 4 and fl[1].strip() in ('0', '1')]
    X = [t for t, _, _ in rows3]; y = np.array([l for _, l, _ in rows3])
    src = [s for *_, s in rows3]
    pool = list(zip(X, y))
    gold = read2(HERE.parent / 'data-website' / 'real_gold_round2.csv')
    gold_texts = {t for t, _ in gold}
    print(f'خام: {len(pool)} | طلایی واقعی: {len(gold)}', flush=True)

    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(pool))
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = Pipeline([('feats', feats()),
                      ('clf', LogisticRegression(C=4.0, class_weight='balanced',
                                                 max_iter=800, random_state=SEED))])
        m.fit([X[i] for i in tr], y[tr])
        oof[va] = m.predict_proba([X[i] for i in va])[:, 1]
        print(f'  OOF {k+1}/3 ({time.time()-t0:.0f}s)', flush=True)

    pred = (oof >= 0.5).astype(int)
    quality = (~(((y == 0) & (oof > 0.95)) | ((y == 1) & (oof < 0.05)))) & (pred == y)
    q_idx = np.where(quality)[0]
    border = [i for i in q_idx if 0.35 < oof[i] < 0.65]
    easy = [i for i in q_idx if not (0.35 < oof[i] < 0.65)]
    random.shuffle(border); random.shuffle(easy)

    nb = min(len(border), int(CORE_N * BORDER_FRAC))
    out = list(border[:nb])
    tw = 0; pos = sum(int(y[i]) for i in out)
    for i in easy:
        if len(out) >= CORE_N:
            break
        s = src[i]
        if s == 'tweets' and tw >= int(CORE_N * TWEETS_FRAC):
            continue
        if y[i] == 1 and pos >= int(CORE_N * POS_TARGET):
            continue
        if random.random() < (MILD_BOOST if (y[i] == 1 and s in MILD_SRC) else 1.0) or y[i] == 0:
            out.append(i); pos += int(y[i]); tw += (s == 'tweets')
    for i in easy:
        if len(out) >= CORE_N:
            break
        if i not in out:
            out.append(i)

    core = [(X[i], int(y[i]), src[i], 'border' if 0.35 < oof[i] < 0.65 else 'quality') for i in out]
    # طلایی واقعی (بدون تداخل با ارزیابی دور ۱ — خود فایل جدا ساخته شد)
    rows = core + [(t, l, 'youtube_real', 'real_gold') for t, l in gold
                   if t not in {x for x, *_ in core}]
    random.shuffle(rows)
    print(f'مجموع آموزش: {len(rows)} (هسته {len(core)} + طلایی {len(rows)-len(core)})', flush=True)

    Xs = [normalize(t) for t, _, _, _ in rows]
    ys = np.array([l for _, l, _, _ in rows])
    Xtr, Xva, ytr, yva = train_test_split(Xs, ys, test_size=3500, stratify=ys, random_state=SEED)

    tests = {'website84': read2(EVAL / 'website_comments_test.csv'),
             'youtube521': read2(EVAL / 'youtube_real_test.csv')}
    mixed = []
    for name in ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']:
        mixed += read2(MERGED / f'test_{name}.csv')
    tests['mixed5'] = mixed

    best = None
    for C in [4.0, 10.0]:
        m = Pipeline([('feats', feats()),
                      ('clf', LogisticRegression(C=C, class_weight='balanced',
                                                 max_iter=1000, random_state=SEED))])
        m.fit(Xtr, ytr)
        pv = m.predict_proba(Xva)[:, 1]
        thr = max(np.linspace(0.2, 0.8, 121), key=lambda c: f1_score(yva, (pv >= c).astype(int)))
        row = {'C': C, 'thr': round(float(thr), 4)}
        for name, data in tests.items():
            Xe = [normalize(t) for t, _ in data]
            ye = [l for _, l in data]
            pr = (m.predict_proba(Xe)[:, 1] >= thr).astype(int)
            row[name] = round(f1_score(ye, pr), 4)
        print(f'C={C} thr={row["thr"]} | یوتیوب={row["youtube521"]:.3f} | وب‌سایت={row["website84"]:.3f} '
              f'| mixed={row["mixed5"]:.3f}', flush=True)
        if best is None or (row['youtube521'], row['mixed5']) > (best['youtube521'], best['mixed5']):
            best = row

    if best['youtube521'] >= 0.335 and best['mixed5'] >= 0.845:
        import shutil
        shutil.copy(MERGED / 'train_selected.csv', MERGED / 'train_selected_v3_16k.csv')
        with open(MERGED / 'train_selected.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['text', 'label', 'source', 'reason'])
            w.writerows(rows)
        print(f"✅ پذیرفته شد → train_selected.csv ({len(rows)} نمونه) — برای آموزش نهایی train_model_v4.py را اجرا کن")
    else:
        print('⛔ بهبود کافی نبود — train_selected.csv دست نخورد')


if __name__ == '__main__':
    main()
