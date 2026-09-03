# -*- coding: utf-8 -*-
"""آموزش مجدد مدل v2 از طریق وب — با گزارش پیشرفت در train_status.json.

مراحل: load (خواندن داده) → train (آموزش LR C=4) → eval (۵ تست) → save/done
داده اضافی: moderation/extra_data/*.csv با ستون‌های text,label
"""
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import normalize, dual_form  # noqa: E402

MERGED = HERE.parent / 'data-merged'
EXTRA = HERE / 'extra_data'
STATUS = HERE / 'train_status.json'
TESTS = ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']
HEARTBEAT = {'t': 0.0}


def write_status(stage, pct, message, running=True, **kw):
    st = {'stage': stage, 'pct': int(pct), 'message': message, 'running': running,
          'heartbeat': time.time(), 'updated': time.strftime('%H:%M:%S')}
    st.update(kw)
    json.dump(st, open(STATUS, 'w', encoding='utf-8'), ensure_ascii=False)


def load_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        r = csv.reader(f)
        next(r, None)
        out = []
        for fl in r:
            if len(fl) >= 2 and fl[1].strip() in ('0', '1'):
                out.append((normalize(fl[0])[:250], int(fl[1])))
        return out


def main():
    t0 = time.time()
    try:
        write_status('load', 5, 'خواندن داده‌های آموزش...')
        train = load_csv(MERGED / 'train_merged.csv')
        extra_files = sorted(EXTRA.glob('*.csv'))
        extra = []
        for p in extra_files:
            extra += load_csv(p)
        if extra:
            seen = set(t for t, _ in train)
            train += [(t, l) for t, l in extra if t not in seen and not seen.add(t)]
        tests = {n: load_csv(MERGED / f'test_{n}.csv') for n in TESTS
                 if (MERGED / f'test_{n}.csv').exists()}
        pos = sum(l for _, l in train)
        write_status('load', 20, f'{len(train)} نمونه ({pos} مثبت، {len(extra)} داده اضافی از {len(extra_files)} فایل)')

        def union_feats():
            char = Pipeline([('dual', FunctionTransformer(dual_form)),
                             ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                                       min_df=4, max_features=250000,
                                                       sublinear_tf=True, dtype=np.float32))])
            word = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\S+',
                                   ngram_range=(1, 2), min_df=4, sublinear_tf=True, dtype=np.float32)
            return FeatureUnion([('char', char), ('word', word)])

        write_status('train', 30, 'آموزش مدل (LR C=4)...')
        X = [t for t, _ in train]
        y = np.array([l for _, l in train])
        model = Pipeline([('feats', union_feats()),
                          ('clf', LogisticRegression(C=4.0, class_weight='balanced',
                                                     max_iter=800, random_state=42))]).fit(X, y)
        write_status('train', 70, 'آموزش تمام شد — در حال ارزیابی...')

        summary = {}
        for i, (n, pairs) in enumerate(tests.items()):
            xt = [t for t, _ in pairs]
            yt = np.array([l for _, l in pairs])
            pred = (model.predict_proba(xt)[:, 1] >= 0.5).astype(int)
            summary[n] = {
                'P': round(precision_score(yt, pred, pos_label=1, zero_division=0), 3),
                'R': round(recall_score(yt, pred, pos_label=1, zero_division=0), 3),
                'F1': round(f1_score(yt, pred, pos_label=1, zero_division=0), 3),
                'n': len(pairs)}
            write_status('eval', 70 + (i + 1) * 5, f'ارزیابی {n}: F1={summary[n]["F1"]}')

        write_status('save', 92, 'ذخیره مدل...')
        joblib.dump(model, HERE / 'model.joblib')
        write_status('done', 100, f'آموزش کامل شد ({time.time()-t0:.0f} ثانیه)', running=False,
                     summary=summary, train_size=len(train))
    except Exception as e:
        import traceback
        traceback.print_exc()
        write_status('error', 0, f'خطا: {e}', running=False)


if __name__ == '__main__':
    main()
