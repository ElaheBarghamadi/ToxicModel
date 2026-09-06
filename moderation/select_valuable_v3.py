# -*- coding: utf-8 -*-
"""گزینش کیفیت‌محور v3 — «کم اما همہ باکیفیت» + جستجوی بهترین اندازه.

فلسفه (دستور کاربر): تعداد کمتر، همه نمونه‌ها باکیفیت.
قواعد کیفیت:
  ۱) حذف برچسب‌خراب (تقریب Confident Learning با OOF)
  ۲) حذف «همه‌ی» نمونه‌های تناقض‌دار (مدلِ فولدهای دیگر با هر فاصله‌ای مخالف برچسب رأی داده) — برخلاف v2 که سقف ۱۲۰۰تای سخت نگه می‌داشت
  ۳) فقط نمونه‌های تاییدشده: برچسب == پیش‌بینی OOF (مرزی‌های درست تا ۱۰٪ نگه داشته می‌شوند — گرچه مرزی‌اند، برچسبشان صحیح و ارزش مرزی دارند)
  ۴) حذف تکراری‌های نرمال‌شده
جستجو: اندازه‌های ۸k/۱۲k/۱۶k + خط پایه ۱۹٫۸k-v2 → ارزیابی روی val + وب‌سایت ۸۴ + یوتیوب واقعی ۵۲۱ + ۵ تست خارجی.
خروجی برنده: train_selected.csv (نسخه v2 → train_selected_v2.csv)
"""
import csv, json, random, re, shutil, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mod_text import dual_form, normalize

MERGED = HERE.parent / 'data-merged'
EVAL = HERE.parent / 'eval'
SIZES = [8000, 12000, 16000]
BORDER_FRAC = 0.10      # سهم مرزی‌های درست
MILD_SRC = {'phicad', 'phate', 'naseza'}
MILD_BOOST = 1.6
TWEETS_FRAC = 0.20      # سهمیه توییتر نسبت به اندازه
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


def read_csv(p):
    with open(p, encoding='utf-8') as f:
        r = csv.reader(f)
        next(r)
        return [(fl[0], int(fl[1])) for fl in r if len(fl) >= 2 and len(fl[0]) >= 4]


def eval_f1(model, thr, data):
    X = [normalize(t) for t, _ in data]
    y = [l for _, l in data]
    p = model.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    return f1_score(y, pred), precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0)


def best_thr(model, Xv, yv):
    pv = model.predict_proba(Xv)[:, 1]
    cs = np.linspace(0.2, 0.8, 121)
    return max(cs, key=lambda c: f1_score(yv, (pv >= c).astype(int)))


def main():
    t0 = time.time()
    pool = read_csv(MERGED / 'train_merged.csv')
    X = [t for t, _ in pool]
    y = np.array([l for _, l in pool])
    src = [r[2] for r in [None] * 0]  # placeholder
    with open(MERGED / 'train_merged.csv', encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        src = [fl[2] for fl in r if len(fl) >= 3 and len(fl[0]) >= 4]
    print(f'خام: {len(pool)}', flush=True)

    # ---------- OOF یک‌بار ----------
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(pool))
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = Pipeline([('feats', feats()),
                      ('clf', LogisticRegression(C=4.0, class_weight='balanced',
                                                 max_iter=800, random_state=SEED))])
        m.fit([X[i] for i in tr], y[tr])
        oof[va] = m.predict_proba([X[i] for i in va])[:, 1]
        print(f'  OOF فولد {k+1}/3 ({time.time()-t0:.0f}s)', flush=True)

    pred = (oof >= 0.5).astype(int)
    suspect = ((y == 0) & (oof > 0.95)) | ((y == 1) & (oof < 0.05))
    quality = (~suspect) & (pred == y)          # ← قلب v3: فقط تاییدشده‌ها
    print(f'مشکوک به برچسب غلط: {int(suspect.sum())} | تناقض‌دار حذف‌شده: {int((pred != y).sum())} '
          f'| باکیفیت باقی‌مانده: {int(quality.sum())}', flush=True)

    q_idx = np.where(quality)[0]
    border = [i for i in q_idx if 0.35 < oof[i] < 0.65]
    easy = [i for i in q_idx if not (0.35 < oof[i] < 0.65)]
    random.shuffle(border); random.shuffle(easy)

    # حذف تکراری نرمال‌شده
    def norm_key(t):
        return re.sub(r'\W+', '', normalize(t))[:80]
    seen = set()
    easy_u = []
    for i in easy:
        k = norm_key(X[i])
        if k not in seen:
            seen.add(k); easy_u.append(i)
    easy = easy_u
    print(f'بعد از حذف تکراری: easy={len(easy)} border={len(border)}', flush=True)

    # ---------- مجموعه‌های آزمایشی ----------
    def build(n):
        nb = min(len(border), int(n * BORDER_FRAC))
        out = list(border[:nb])
        tw = 0
        pos = sum(y[i] for i in out)
        rest = []
        for i in easy:
            if len(out) >= n:
                break
            s = src[i]
            if s == 'tweets' and tw >= int(n * TWEETS_FRAC):
                continue
            if y[i] == 1 and pos >= int(n * POS_TARGET):
                continue
            w = (MILD_BOOST if (y[i] == 1 and s in MILD_SRC) else 1.0)
            if random.random() < w or y[i] == 0:
                out.append(i); rest.append(i)
                pos += int(y[i])
                tw += (s == 'tweets')
        # اگر کم آمد، بدون قید مثبت تکمیل
        for i in easy:
            if len(out) >= n:
                break
            if i not in out:
                out.append(i)
        random.shuffle(out)
        return out

    tests = {
        'website84': read_csv(EVAL / 'website_comments_test.csv'),
        'youtube521': read_csv(EVAL / 'youtube_real_test.csv'),
    }
    mixed = []
    for name in ['tweets', 'naseza', 'pars_offensive', 'phate', 'phicad']:
        mixed += read_csv(MERGED / f'test_{name}.csv')
    tests['mixed5'] = mixed

    results = {}
    for n in SIZES:
        idx = build(n)
        Xs = [X[i] for i in idx]; ys = np.array([y[i] for i in idx])
        Xtr, Xva, ytr, yva = train_test_split(Xs, ys, test_size=3500,
                                              stratify=ys, random_state=SEED)
        m = Pipeline([('feats', feats()),
                      ('clf', LogisticRegression(C=4.0, class_weight='balanced',
                                                 max_iter=800, random_state=SEED))])
        m.fit([normalize(t) for t in Xtr], ytr)
        thr = best_thr(m, [normalize(t) for t in Xva], yva)
        vf1 = f1_score(yva, (m.predict_proba([normalize(t) for t in Xva])[:, 1] >= thr).astype(int))
        row = {'n': n, 'pos': f'{ys.mean():.0%}', 'thr': round(float(thr), 3), 'val_F1': round(vf1, 4)}
        for name, data in tests.items():
            f1, p, r = eval_f1(m, thr, data)
            row[name] = round(f1, 4)
        results[n] = row
        print(f'{n:>6} | ع={row["pos"]} thr={row["thr"]} | val={row["val_F1"]:.3f} '
              f'| وب‌سایت={row["website84"]:.3f} | یوتیوب={row["youtube521"]:.3f} '
              f'| ۵تستی={row["mixed5"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
        # ذخیره هر اندازه برای برنده
        with open(MERGED / f'train_selected_cand_{n}.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['text', 'label', 'source', 'reason'])
            for i in idx:
                rsn = 'border' if 0.35 < oof[i] < 0.65 else 'quality'
                w.writerow([X[i], int(y[i]), src[i], rsn])

    json.dump({'sizes': {str(k): v for k, v in results.items()},
               'pool_quality': int(quality.sum())},
              open(HERE / 'v3_sweep.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('✅ sweep ذخیره شد → moderation/v3_sweep.json')


if __name__ == '__main__':
    main()
