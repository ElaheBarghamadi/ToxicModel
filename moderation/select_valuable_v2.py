# -*- coding: utf-8 -*-
"""گزینش ارزش‌محور نسخه ۲ — «بهترین داده» در همان بودجه ۱۹٬۸۰۰.

بهبودها نسبت به v1 (با کتابچه证据 قبلی):
  ۱) پالایش برچسب‌خراب: حذف نمونه‌هایی که مدلِ آموزش‌دیده در فولدهای دیگر با اطمینان
     بالا مخالف برچسب‌شان رأی می‌دهد (تقریب Confident Learning):
       label=0 با OOF>0.95 یا label=1 با OOF<0.05 → مشکوک به برچسب غلط → حذف
  ۲) غنی‌سازی دسته‌های ضعیف: نمونه‌های مثبتِ منابع «توهین ملایم» (phicad/phate/naseza)
     با ضریب ۱٫۶ برابر در پرکننده
  ۳) سهمیه توییتر ۴٬۵۰۰ (تک‌مدل است؛ حفظ تعادل)
  ۴) ۱٬۲۰۰ سخت + ۱٬۲۰۰ مرزی (کمی بیشتر از قبل، نه زیاد که کالیبراسیون خراب شود)

خروجی: train_selected.csv (بازنویسی) — قبلی در train_selected_v1.csv بایگانی می‌شود
"""
import csv, random, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import dual_form

MERGED = HERE.parent / 'data-merged'
BUDGET = 19800
HARD_CAP, BORDER_CAP = 1200, 1200
TWEETS_QUOTA = 4500
MILD_SRC = {'phicad', 'phate', 'naseza'}
MILD_BOOST = 1.6
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


def main():
    t0 = time.time()
    rows = []
    with open(MERGED / 'train_merged.csv', encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        for fl in r:
            if len(fl) >= 3 and len(fl[0]) >= 4:
                rows.append((fl[0], int(fl[1]), fl[2]))
    X = [t for t, _, _ in rows]; y = np.array([l for _, l, _ in rows]); src = [s for _, _, s in rows]
    print(f'خام: {len(rows)}', flush=True)

    # ---------- OOF ----------
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(rows))
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = Pipeline([('feats', feats()),
                      ('clf', LogisticRegression(C=4.0, class_weight='balanced',
                                                 max_iter=800, random_state=SEED))])
        m.fit([X[i] for i in tr], y[tr])
        oof[va] = m.predict_proba([X[i] for i in va])[:, 1]
        print(f'  فولد {k+1}/3 ({time.time()-t0:.0f}s)', flush=True)

    # ---------- ۱) حذف برچسب‌خراب ----------
    suspect = ((y == 0) & (oof > 0.95)) | ((y == 1) & (oof < 0.05))
    print(f'نمونه‌های مشکوک به برچسب غلط (حذف): {int(suspect.sum())} '
          f'({suspect.mean():.1%})', flush=True)
    keep = ~suspect
    Xk = [X[i] for i in range(len(X)) if keep[i]]
    yk = y[keep]; srck = [src[i] for i in range(len(X)) if keep[i]]
    oofk = oof[keep]

    pred = (oofk >= 0.5).astype(int)
    correct = pred == yk
    margin = np.abs(oofk - yk)
    hard_idx = np.where(~correct)[0]
    border_idx = np.where(correct & (oofk > 0.35) & (oofk < 0.65))[0]
    easy_idx = np.where(correct & ~((oofk > 0.35) & (oofk < 0.65)))[0]

    hard = [(Xk[i], int(yk[i]), srck[i]) for i in hard_idx[np.argsort(-margin[hard_idx])][:HARD_CAP]]
    border = [(Xk[i], int(yk[i]), srck[i])
              for i in border_idx[np.argsort(-np.abs(oofk[border_idx] - 0.5))][:BORDER_CAP]]

    # ---------- ۲) پرکننده غنی‌شده ----------
    used = {t for t, _, _ in hard} | {t for t, _, _ in border}
    pool = [(Xk[i], int(yk[i]), srck[i]) for i in easy_idx if Xk[i] not in used]
    random.shuffle(pool)

    def weight(item):
        t, l, s = item
        w = 1.0
        if l == 1 and s in MILD_SRC:
            w *= MILD_BOOST     # توهین ملایم — دسته ضعیف
        if s == 'tweets':
            w *= 0.8            # توییتر کمتر (تک‌مدل؛ سهمیه جدا هم داریم)
        return w

    tw_count = 0
    pos_target = int(BUDGET * 0.50)
    pos_count = sum(1 for _, l, _ in hard + border if l == 1)
    fill = []
    # اول وزن‌دار از کل استخر، بعد تکمیل
    weighted = sorted(pool, key=lambda it: -random.random() ** (1 / weight(it)))
    seen_src = Counter()
    out = []
    for t, l, s in weighted:
        if len(out) >= BUDGET - len(hard) - len(border):
            break
        if s == 'tweets' and tw_count >= TWEETS_QUOTA:
            continue
        if l == 1 and pos_count >= pos_target:
            continue
        out.append((t, l, s))
        seen_src[s] += 1
        if s == 'tweets':
            tw_count += 1
        if l == 1:
            pos_count += 1
    # اگر هنوز جا مانده، از باقی بدون قید مثبت
    rest = [it for it in pool if it not in out]
    for t, l, s in rest:
        if len(out) >= BUDGET - len(hard) - len(border):
            break
        if s == 'tweets' and tw_count >= TWEETS_QUOTA:
            continue
        out.append((t, l, s))
        if s == 'tweets':
            tw_count += 1

    sel = [(t, l, s, 'hard') for t, l, s in hard] + \
          [(t, l, s, 'border') for t, l, s in border] + \
          [(t, l, s, 'easy') for t, l, s in out]
    seen = set(); dedup = []
    for t, l, s, rs in sel:
        if t not in seen:
            seen.add(t); dedup.append((t, l, s, rs))
    sel = dedup[:BUDGET]
    random.shuffle(sel)

    shutil.copy(MERGED / 'train_selected.csv', MERGED / 'train_selected_v1.csv')
    with open(MERGED / 'train_selected.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'label', 'source', 'reason'])
        w.writerows(sel)

    pos = sum(l for _, l, _, _ in sel)
    print(f'\n✅ {len(sel)} نمونه | مثبت {pos} ({pos/len(sel):.1%}) | '
          f'توییتر {tw_count} | ارزش: {dict(Counter(r for *_, r in sel))}')
    for s, c in Counter(s for _, _, s, _ in sel).most_common():
        p = sum(1 for _, l, ss, _ in sel if ss == s and l == 1)
        print(f'  {s:16s} {c:6d} ({c/len(sel):.0%}) | مثبت {p}')
    print(f'زمان: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
