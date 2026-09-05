# -*- coding: utf-8 -*-
"""آموزش مجدد از طریق وب — نسخه ۴ (تک‌مدل لجستیک، بدون بازبینی).

ایمن‌سازی‌ها:
  ۱) داده اضافه آپلودی در برابر فایل‌های تست dedup می‌شود
  ۲) قبل از بازنویسی، مدل در models_archive/ بایگانی می‌شود (۳ نسخه)
  ۳) سنجش سلامت: F1 تست توییتی < 0.45 → رد و حفظ مدل قبلی
  ۴) آستانه فقط از validation داخلی + تولید گزارش
"""
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import normalize, dual_form  # noqa: E402

MERGED = HERE.parent / 'data-merged'
EXTRA = HERE / 'extra_data'
ARCHIVE = HERE / 'models_archive'
STATUS = HERE / 'train_status.json'
TESTS = ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']
MIN_F1_TWEETS = 0.45
BEST_C = 4.0


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


def feats():
    char = Pipeline([('dual', FunctionTransformer(dual_form)),
                     ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                               min_df=3, max_features=200000,
                                               sublinear_tf=True, dtype=np.float32))])
    word = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\S+',
                           ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([('char', char), ('word', word)])


def logreg(C=BEST_C):
    return Pipeline([('feats', feats()),
                     ('clf', LogisticRegression(C=C, class_weight='balanced',
                                                max_iter=1000, random_state=42))])


def main():
    t0 = time.time()
    try:
        write_status('load', 5, 'خواندن داده‌ها...')
        sel_path = MERGED / 'train_selected.csv'
        train = load_csv(sel_path) if sel_path.exists() else load_csv(MERGED / 'train_merged.csv')
        tests = {n: load_csv(MERGED / f'test_{n}.csv') for n in TESTS
                 if (MERGED / f'test_{n}.csv').exists()}
        test_texts = {t for v in tests.values() for t, _ in v}

        extra_files = sorted(EXTRA.glob('*.csv'))
        extra_raw = []
        for p in extra_files:
            extra_raw += load_csv(p)
        extra = [(t, l) for t, l in extra_raw if t not in test_texts]
        dropped = len(extra_raw) - len(extra)
        train = train + extra
        write_status('load', 15, f'{len(train)} نمونه (پالایش نشت: {dropped} مورد حذف)')

        write_status('train', 25, 'تنظیم آستانه روی validation...')
        tr, val = train_test_split(train, test_size=3000,
                                   stratify=[l for _, l in train], random_state=42)
        mt = logreg().fit([t for t, _ in tr], [l for _, l in tr])
        pv = mt.predict_proba([t for t, _ in val])[:, 1]
        yv = np.array([l for _, l in val])
        prec, rec, thr = precision_recall_curve(yv, pv)
        f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        i = int(np.argmax(f1s[:-1]))
        rt = float(np.clip(thr[i], 0.3, 0.7))
        del mt
        write_status('train', 45, f'آستانه بلاک از validation: {rt:.3f}')

        write_status('train', 55, 'آموزش نهایی LogisticRegression...')
        model = logreg().fit([t for t, _ in train], [l for _, l in train])
        write_status('train', 80, 'ارزیابی سلامت...')

        xt = [t for t, _ in tests['tweets']]; yt = np.array([l for _, l in tests['tweets']])
        f1_tw = f1_score(yt, (model.predict_proba(xt)[:, 1] >= rt).astype(int), zero_division=0)
        if f1_tw < MIN_F1_TWEETS:
            write_status('error', 0, f'رد شد: F1 توییتی {f1_tw:.3f} < {MIN_F1_TWEETS} — مدل قبلی حفظ شد',
                         running=False)
            return

        summary = {}
        for i2, (n, pairs) in enumerate(tests.items()):
            xx = [t for t, _ in pairs]; yy = np.array([l for _, l in pairs])
            pred = (model.predict_proba(xx)[:, 1] >= rt).astype(int)
            summary[n] = {'P': round(precision_score(yy, pred, pos_label=1, zero_division=0), 3),
                          'R': round(recall_score(yy, pred, pos_label=1, zero_division=0), 3),
                          'F1': round(f1_score(yy, pred, pos_label=1, zero_division=0), 3),
                          'n': len(pairs)}
            write_status('eval', 80 + i2 * 4, f'ارزیابی {n}: F1={summary[n]["F1"]}')

        write_status('save', 92, 'بایگانی نسخه قبلی و ذخیره...')
        ARCHIVE.mkdir(exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        if (HERE / 'model.joblib').exists():
            shutil.copy(HERE / 'model.joblib', ARCHIVE / f'model_{ts}.joblib')
            olds = sorted(ARCHIVE.glob('model_*.joblib'))
            for old in olds[:-3]:
                old.unlink()
        joblib.dump(model, HERE / 'model.joblib')
        cfg = {'version': f'web-{ts}', 'algo': f'LogisticRegression (C={BEST_C})',
               'block_threshold': round(rt, 4), 'selected_on': 'internal validation (3k)',
               'train_size': len(train), 'extra_added': len(extra),
               'extra_dropped_leak': dropped, 'trained_at': time.strftime('%Y-%m-%d %H:%M'),
               'decisions': ['ok', 'block']}
        json.dump(cfg, open(HERE / 'model_config.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        try:
            write_status('save', 97, 'تولید گزارش...')
            from report_utils import generate as gen_report
            gen_report()
        except Exception:
            import traceback
            traceback.print_exc()
        write_status('done', 100, f'کامل شد ({time.time()-t0:.0f} ثانیه) — F1 توییتی {f1_tw:.3f}',
                     running=False, summary=summary, config=cfg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        write_status('error', 0, f'خطا: {e}', running=False)


if __name__ == '__main__':
    main()
