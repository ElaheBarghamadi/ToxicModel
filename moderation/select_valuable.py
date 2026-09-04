# -*- coding: utf-8 -*-
"""گزینش ارزش‌محور داده — کاهش مجموعه آموزش به زیر ۲۰ هزار نمونه (۱۹٬۸۰۰).

روش (انتخاب ترکیب بر اساس آزمایش روی validation ثابت ۵٬۴۷۰ نمونه —
ترکیب‌های A/B/C اختلاف معنادار نداشتند و C انتخاب شد):

  ۱) آموزش متقاطع ۳-فولدی روی کل داده → احتمال out-of-fold هر نمونه
  ۲) دسته‌بندی: hard (پیش‌بینی غلط) / border (احتمال ۰٫۳۵-۰٫۶۵) / easy (بقیه)
  ۳) ترکیب نهایی:
     • ۱٬۰۰۰ سخت‌ترین (hard)  — بیش از این، برچسب‌نویزی غالب می‌شود و کالیبراسیون خراب می‌شود
     • ۱٬۰۰۰ مرزی‌ترین (border)
     • ۱۷٬۸۰۰ آسان به‌صورت طبقه‌بندی‌شده متناسب با (منبع، برچسب) + سهمیه توییتر ۵٬۰۰۰
       (سهمیه توییتر برای قوی‌ماندن مدل فحش v1)

خروجی: data-merged/train_selected.csv با ستون reason (hard/border/easy)
"""
import csv
import random
import sys
import time
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
from mod_text import dual_form  # noqa: E402

MERGED = HERE.parent / 'data-merged'
BUDGET = 19800
HARD_CAP = 1000
BORDER_CAP = 1000
TWEETS_QUOTA = 5000
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


def stratified(pool, n):
    """نمونه‌گیری متناسب با (منبع، برچسب) + سهمیه توییتر + سقف ۵۰٪ برای هر منبع."""
    by_cs = defaultdict(list)
    for t, l, s in pool:
        by_cs[(s, l)].append((t, l, s))
    src_total = Counter(s for s, _ in by_cs for _ in by_cs[(s, _)])
    tw_total = src_total.get('tweets', 0)
    out = []
    for (s, l), items in by_cs.items():
        share = n * len(items) / len(pool)
        if s == 'tweets' and tw_total > 0:
            share = min(share, n * TWEETS_QUOTA / tw_total)
        if s != 'tweets':
            share = min(share, n * 0.50 * len(items) / src_total[s])
        random.shuffle(items)
        out += [(t, l, s) for t, l, s in items[:int(share)]]
    used = {t for t, _, _ in out}
    cands = [(t, l, s) for t, l, s in pool if t not in used]
    random.shuffle(cands)
    for t, l, s in cands:
        if len(out) >= n:
            break
        out.append((t, l, s))
    return out[:n]


def main():
    t0 = time.time()
    rows = []
    with open(MERGED / 'train_merged.csv', encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        for fl in r:
            if len(fl) >= 3 and len(fl[0]) >= 4:
                rows.append((fl[0], int(fl[1]), fl[2]))
    print(f'خام: {len(rows)} نمونه', flush=True)

    X = [t for t, _, _ in rows]
    y = np.array([l for _, l, _ in rows])
    src = [s for _, _, s in rows]

    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(rows))
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = Pipeline([('feats', feats()),
                      ('clf', LogisticRegression(C=4.0, class_weight='balanced',
                                                 max_iter=800, random_state=SEED))])
        m.fit([X[i] for i in tr], y[tr])
        oof[va] = m.predict_proba([X[i] for i in va])[:, 1]
        print(f'  فولد {k+1}/3 ({time.time()-t0:.0f}s)', flush=True)

    pred = (oof >= 0.5).astype(int)
    correct = pred == y
    margin = np.abs(oof - y)
    hard_idx = np.where(~correct)[0]
    border_idx = np.where(correct & (oof > 0.35) & (oof < 0.65))[0]
    easy_pool = [(X[i], y[i], src[i]) for i in np.where(correct & ~((oof > 0.35) & (oof < 0.65)))[0]]

    hard_sel = [(X[i], y[i], src[i]) for i in hard_idx[np.argsort(-margin[hard_idx])][:HARD_CAP]]
    border_sel = [(X[i], y[i], src[i])
                  for i in border_idx[np.argsort(-np.abs(oof[border_idx] - 0.5))][:BORDER_CAP]]
    fill = stratified(easy_pool, BUDGET - len(hard_sel) - len(border_sel))

    sel = [(t, l, s, 'hard') for t, l, s in hard_sel] + \
          [(t, l, s, 'border') for t, l, s in border_sel] + \
          [(t, l, s, 'easy') for t, l, s in fill]
    seen, dedup = set(), []
    for t, l, s, rs in sel:
        if t not in seen:
            seen.add(t); dedup.append((t, l, s, rs))
    sel = dedup[:BUDGET]
    random.shuffle(sel)

    with open(MERGED / 'train_selected.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'label', 'source', 'reason'])
        w.writerows(sel)

    cats = Counter(rs for *_, rs in sel)
    pos = sum(l for _, l, _, _ in sel)
    print(f'\n✅ train_selected.csv: {len(sel)} نمونه | مثبت {pos} ({pos/len(sel):.1%})')
    print('  ارزش:', dict(cats))
    for s, c in Counter(s for _, _, s, _ in sel).most_common():
        print(f'  {s:16s} {c:6d} ({c/len(sel):.0%})')
    print(f'زمان: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
