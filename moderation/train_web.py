# -*- coding: utf-8 -*-
"""آموزش مجدد از طریق وب — نسخه ۳ (الزام لجستیک + ایمن‌سازی).

ایمن‌سازی‌های جدید:
  ۱) داده اضافه‌ی آپلودی در برابر «همه‌ی فایل‌های تست» dedup می‌شود (ضد نشت/آلودگی)
  ۲) قبل از بازنویسی، مدل فعلی در models_archive/ بایگانی می‌شود (نگه‌داری ۳ نسخه)
  ۳) سنجش سلامت: اگر F1 تست توییتی < 0.45 باشد آموزش «رد» می‌شود و مدل قدیمی می‌ماند
  ۴) آستانه‌ها روی validation داخلی انتخاب می‌شوند و در model_config.json ذخیره می‌شوند

مدل: LogisticRegression (هر دو) — مطابق الزام تکلیف.
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
                                               min_df=4, max_features=250000,
                                               sublinear_tf=True, dtype=np.float32))])
    word = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\S+',
                           ngram_range=(1, 2), min_df=4, sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([('char', char), ('word', word)])


def logreg(C):
    return Pipeline([('feats', feats()),
                     ('clf', LogisticRegression(C=C, class_weight='balanced',
                                                max_iter=1000, random_state=42))])


def pick_thresholds(m1, m2, Xval, yval):
    """انتخاب آستانه‌ها فقط روی validation."""
    p1 = m1.predict_proba(Xval)[:, 1]
    p2 = m2.predict_proba(Xval)[:, 1]
    prec, rec, thr = precision_recall_curve(yval, p2)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    review_t = float(np.clip(thr[int(np.argmax(f1s[:-1]))], 0.3, 0.7))
    best = None
    for t1 in (0.80, 0.85, 0.90, 0.95):
        for t2 in (0.5, 0.6, 0.7, 0.8, 0.9):
            b = (p1 >= t1) & (p2 >= t2)
            if b.sum() < 30:
                continue
            Pb = precision_score(yval, b, zero_division=0)
            if Pb >= 0.97:
                Rb = recall_score(yval, b)
                if best is None or Rb > best[0]:
                    best = (Rb, t1, t2)
    t1, t2 = (best[1], best[2]) if best else (0.8, 0.9)
    return review_t, t1, t2


def main():
    t0 = time.time()
    try:
        write_status('load', 5, 'خواندن داده‌ها...')
        train2 = load_csv(MERGED / 'train_merged.csv')
        train1 = load_csv(HERE.parent / 'persian-abusive-words' / 'clean' / 'train_clean.csv')
        tests = {n: load_csv(MERGED / f'test_{n}.csv') for n in TESTS
                 if (MERGED / f'test_{n}.csv').exists()}
        test_texts = {t for v in tests.values() for t, _ in v}

        # (ایمن‌سازی ۱) حذف داده اضافی هم‌پوشان با تست‌ها
        extra_files = sorted(EXTRA.glob('*.csv'))
        extra_raw = []
        for p in extra_files:
            extra_raw += load_csv(p)
        before = len(extra_raw)
        extra = [(t, l) for t, l in extra_raw if t not in test_texts]
        dropped = before - len(extra)
        train2 = train2 + extra
        write_status('load', 15, f'{len(train2)} نمونه (اسپم‌گیری داده اضافی: {dropped} مورد حذف شد)')

        # validation داخلی برای آستانه‌ها
        write_status('train', 20, 'آموزش مدل‌های آزمایشی برای تنظیم آستانه...')
        tr2, val2 = train_test_split(train2, test_size=6000,
                                     stratify=[l for _, l in train2], random_state=42)
        m1t = logreg(10.0).fit([t for t, _ in train1], [l for _, l in train1])
        m2t = logreg(4.0).fit([t for t, _ in tr2], [l for _, l in tr2])
        rt, bt1, bt2 = pick_thresholds(m1t, m2t, [t for t, _ in val2],
                                       np.array([l for _, l in val2]))
        write_status('train', 45, f'آستانه‌ها از validation: review={rt:.3f}، block=(v1≥{bt1}، v2≥{bt2})')
        del m1t, m2t

        write_status('train', 50, 'آموزش نهایی LogisticRegression...')
        m1 = logreg(10.0).fit([t for t, _ in train1], [l for _, l in train1])
        m2 = logreg(4.0).fit([t for t, _ in train2], [l for _, l in train2])
        write_status('train', 80, 'آموزش تمام شد — ارزیابی سلامت...')

        # (ایمن‌سازی ۳) سنجش سلامت قبل از ذخیره
        xt = [t for t, _ in tests['tweets']]; yt = np.array([l for _, l in tests['tweets']])
        f1_tw = f1_score(yt, (m2.predict_proba(xt)[:, 1] >= rt).astype(int), pos_label=1, zero_division=0)
        if f1_tw < MIN_F1_TWEETS:
            write_status('error', 0, f'رد شد: F1 توییتی {f1_tw:.3f} < {MIN_F1_TWEETS} — مدل قبلی حفظ شد',
                         running=False)
            return

        summary = {}
        for i, (n, pairs) in enumerate(tests.items()):
            xx = [t for t, _ in pairs]; yy = np.array([l for _, l in pairs])
            pred = (m2.predict_proba(xx)[:, 1] >= rt).astype(int)
            summary[n] = {'P': round(precision_score(yy, pred, pos_label=1, zero_division=0), 3),
                          'R': round(recall_score(yy, pred, pos_label=1, zero_division=0), 3),
                          'F1': round(f1_score(yy, pred, pos_label=1, zero_division=0), 3),
                          'n': len(pairs)}
            write_status('eval', 80 + (i + 1) * 3, f'ارزیابی {n}: F1={summary[n]["F1"]}')

        # (ایمن‌سازی ۲) بایگانی نسخه قبلی
        write_status('save', 95, 'بایگانی نسخه قبلی و ذخیره...')
        ARCHIVE.mkdir(exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        if (HERE / 'model.joblib').exists():
            shutil.copy(HERE / 'model.joblib', ARCHIVE / f'model_{ts}.joblib')
            olds = sorted(ARCHIVE.glob('model_*.joblib'))
            for old in olds[:-3]:
                old.unlink()
        joblib.dump(m2, HERE / 'model.joblib')
        joblib.dump(m1, HERE / 'model_v1.joblib')
        cfg = {'version': f'web-{ts}', 'trained_at': time.strftime('%Y-%m-%d %H:%M'),
               'review_threshold': round(rt, 4), 'block_v1': bt1, 'block_v2': bt2,
               'selected_on': 'internal validation (6k)', 'algo': 'LogisticRegression',
               'train_size_v2': len(train2), 'extra_rows_added': len(extra),
               'extra_rows_dropped_leak': dropped}
        json.dump(cfg, open(HERE / 'model_config.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        # تولید گزارش کامل (report.json) — اگر شکست خورد، آموزش رد نمی‌شود
        try:
            write_status('save', 97, 'تولید گزارش کامل...')
            sys.path.insert(0, str(HERE))
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
