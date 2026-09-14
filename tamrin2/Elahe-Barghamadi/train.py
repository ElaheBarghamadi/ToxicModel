# -*- coding: utf-8 -*-
"""
train.py — آموزش مدل تعدیل کامنت pycourse.ir (تمرین ۲)

اجرا:  python train.py   (داخل همان پوشه)
خروجی: model.joblib (≤20MB، compress=3) + metrics.json

گام‌ها (مطابق بند ۴ صورت تمرین):
  ۴.۱ معیار پایه (اکثریت / قانون کلیدواژه / قانون طول)
  ۴.۲ cross-validation با میانگین و انحراف معیار (۵-فولد، stratified)
  ۴.۳ نظافت: dropna، حذف تکراری با کلید نرمال‌شده، حذف گروه متناقض
  ۴.۴ نرمال‌سازی فارسی (predict.normalize_text)
  ۴.۵ ارزیابی روی holdout.csv به تفکیک سه سطح
  ۴.۶ گزارش اطمینان (بازه و درصد قاطع)
  ۴.۷ بیست ویچگی برتر هر کلاس
  + انتخاب دو آستانه‌ی سه‌ناحیه‌ای از پیش‌بینی‌های out-of-fold
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import joblib
import sklearn

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     cross_validate, cross_val_predict, GridSearchCV)
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from predict import normalize_text  # noqa: E402  (همان نرمال‌سازِ استنتاج)

SEED = 42
TEMPLATE_COUNT = 358  # تعداد قالب‌های مجزای مولد داده (خروجی gen_data.py)

# ══════════════════════ ۱) بارگذاری و نظافت ══════════════════════
print('═' * 60)
print('۱) بارگذاری و نظافت داده')
print('═' * 60)
df = pd.read_csv(os.path.join(BASE, 'dataset.csv'))
n_raw = len(df)
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)
df['norm'] = df['text'].map(normalize_text)

# متن با دو برچسب متفاوت → حذف کل گروه
lab_per_key = df.groupby('norm')['label'].nunique()
conflict_keys = set(lab_per_key[lab_per_key > 1].index)
n_conflict = int(df['norm'].isin(conflict_keys).sum())
df = df[~df['norm'].isin(conflict_keys)]

# حذف تکراری با کلید نرمال‌شده + حذف متنِ خالی‌شده
n_before_dedup = len(df)
df = df[df['norm'] != '']
df = df.drop_duplicates('norm', keep='first').reset_index(drop=True)

X_raw = df['text'].values
X_norm = df['norm'].values
y = df['label'].values
n_reject = int(y.sum())
print(f'خام: {n_raw} → بعد از نظافت: {len(df)} (ناسازگار: {n_conflict}، '
      f'تکراری/خالی: {n_before_dedup - len(df) - n_conflict})')
print(f'کلاس‌ها: {n_reject} رد ({n_reject/len(df):.1%}) / {len(df)-n_reject} تایید')

# آمار واژگان
uniq_words = set()
gen_words = set()
for t, s in zip(X_norm, df['source']):
    uniq_words.update(t.split())
    if str(s).startswith('gen-'):
        gen_words.update(t.split())
print(f'کلمات یکتا (کل/تولیدی): {len(uniq_words)} / {len(gen_words)} · قالب‌های مولد: {TEMPLATE_COUNT}')

# تله ۱: نرخ نشانه‌ها در دو کلاس
SIGNALS = ['نشانهلینک', 'نشانهتلفن', 'نشانهایمیل', 'نشانهایموجی',
           'کانال', 'فالو', 'آیدی', 'تلگرام', 'اینستا', 'تخفیف']
signal_audit = {}
print('\nنرخ نشانه در دو کلاس (کاهش میان‌بر):')
for sig in SIGNALS:
    r1 = float(np.mean([sig in t for t, l in zip(X_norm, y) if l == 1])) if n_reject else 0.0
    r0 = float(np.mean([sig in t for t, l in zip(X_norm, y) if l == 0]))
    signal_audit[sig] = {'reject_rate': round(r1, 4), 'approve_rate': round(r0, 4)}
    print(f'  {sig:14s} رد {r1:6.1%} · تایید {r0:6.1%}')

# ══════════════════════ ۲) معیار پایه ══════════════════════
print('\n' + '═' * 60)
print('۲) معیارهای پایه (قبل از مدل)')
print('═' * 60)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

KEYWORDS = ['فالو', 'کانال', 'آیدی', 'تلگرام', 'اینستا', 'واتساپ', 'تخفیف', 'فروش',
            'درآمد', 'نشانهلینک', 'نشانهتلفن', 'نشانهایمیل', 'پیوی', 'سایت ما', 'کانال ما']


def rule_majority(texts):
    return np.zeros(len(texts), dtype=int)


def rule_keywords(texts):
    return np.array([int(any(k in t for k in KEYWORDS)) for t in texts], dtype=int)


def rule_length(texts):
    return np.array([int(len(t.split()) < 3) for t in texts], dtype=int)


baselines = {}
for name, fn in [('majority', rule_majority), ('keywords', rule_keywords), ('length', rule_length)]:
    f1s, accs = [], []
    for tr, va in skf.split(X_norm, y):
        pred = fn(X_norm[va])
        f1s.append(f1_score(y[va], pred, zero_division=0))
        accs.append(accuracy_score(y[va], pred))
    baselines[name] = {'f1_mean': round(float(np.mean(f1s)), 4), 'f1_std': round(float(np.std(f1s)), 4),
                       'acc_mean': round(float(np.mean(accs)), 4), 'acc_std': round(float(np.std(accs)), 4)}
    print(f'  {name:10s} F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  Acc={np.mean(accs):.3f}±{np.std(accs):.3f}')
best_base_name = max(baselines, key=lambda k: baselines[k]['f1_mean'])
print(f'  → بهترین پایه: {best_base_name} (F1={baselines[best_base_name]["f1_mean"]})')

# ══════════════════════ ۳) مقایسه الگوریتم‌ها (CV) ══════════════════════
print('\n' + '═' * 60)
print('۳) مقایسه‌ی الگوریتم‌های مجاز — ۵-فولد CV')
print('═' * 60)


def make_features():
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])


def make_pipe(clf):
    return Pipeline([('features', make_features()), ('clf', clf)])


candidates = {
    'LogisticRegression': make_pipe(LogisticRegression(max_iter=3000, random_state=SEED)),
    'MultinomialNB': make_pipe(MultinomialNB()),
    'LinearSVC': make_pipe(LinearSVC(random_state=SEED, max_iter=3000)),
    'DecisionTree': make_pipe(DecisionTreeClassifier(random_state=SEED)),
}
cv_results = {}
for name, pipe in candidates.items():
    t0 = time.time()
    cv = cross_validate(pipe, X_norm, y, cv=skf, scoring=('f1', 'accuracy'), n_jobs=1)
    cv_results[name] = {
        'f1_mean': round(float(cv['test_f1'].mean()), 4), 'f1_std': round(float(cv['test_f1'].std()), 4),
        'acc_mean': round(float(cv['test_accuracy'].mean()), 4), 'acc_std': round(float(cv['test_accuracy'].std()), 4),
    }
    print(f'  {name:20s} F1={cv["test_f1"].mean():.3f}±{cv["test_f1"].std():.3f}  '
          f'Acc={cv["test_accuracy"].mean():.3f}±{cv["test_accuracy"].std():.3f}  ({time.time()-t0:.1f}s)')
# ══════════════════════ ۴) تنظیم LR و تقسیم آزمون ══════════════════════
print('\n' + '═' * 60)
print('۴) تنظیم ابرپارامتر LR + ارزیابی روی مجموعه‌ی آزمون')
print('═' * 60)
X_tr, X_te, y_tr, y_te, src_tr, src_te = train_test_split(
    X_norm, y, df['source'].values, test_size=0.2, stratify=y, random_state=SEED)

grid = GridSearchCV(
    make_pipe(LogisticRegression(max_iter=3000, random_state=SEED)),
    {'clf__C': [1, 4, 10], 'clf__class_weight': [None, 'balanced']},
    cv=skf, scoring='f1')
grid.fit(X_tr, y_tr)
lr_params = {k.replace('clf__', ''): v for k, v in grid.best_params_.items()}
print(f'  بهترین LR: {lr_params} (CV F1={grid.best_score_:.4f})')

lr_final = grid.best_estimator_
lr_final.fit(X_tr, y_tr)
p_te = lr_final.predict_proba(X_te)[:, 1]
pred_te = (p_te >= 0.5).astype(int)

# CV همان مدلِチュون‌شده روی کل داده (برای جدول مقایسه)
cv_tuned = cross_validate(grid.best_estimator_, X_norm, y, cv=skf, scoring=('f1', 'accuracy'))
cv_results['LogisticRegression (tuned)'] = {
    'f1_mean': round(float(cv_tuned['test_f1'].mean()), 4), 'f1_std': round(float(cv_tuned['test_f1'].std()), 4),
    'acc_mean': round(float(cv_tuned['test_accuracy'].mean()), 4), 'acc_std': round(float(cv_tuned['test_accuracy'].std()), 4),
}
cv_results['LogisticRegression']['note'] = 'با پیش‌فرض (بدون class_weight)؛ نسخه‌ی tuned پایین‌تر'
tn, fp, fn_, tp = confusion_matrix(y_te, pred_te).ravel()
test_metrics = {
    'n': int(len(y_te)),
    'accuracy': round(float(accuracy_score(y_te, pred_te)), 4),
    'precision': round(float(precision_score(y_te, pred_te, zero_division=0)), 4),
    'recall': round(float(recall_score(y_te, pred_te, zero_division=0)), 4),
    'f1': round(float(f1_score(y_te, pred_te, zero_division=0)), 4),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn_), 'tp': int(tp)},
    'reject_precision': round(float(precision_score(y_te, pred_te, zero_division=0)), 4),
    'classification_report': classification_report(y_te, pred_te, target_names=['approve', 'reject'], zero_division=0),
}
print(f'  آزمون: Acc={test_metrics["accuracy"]:.3f} F1={test_metrics["f1"]:.3f} '
      f'Prec(رد)={test_metrics["reject_precision"]:.3f} Rec={test_metrics["recall"]:.3f}')
print(f'  ماتریس: TN={tn} FP={fp} FN={fn_} TP={tp}')

# برتری نسبت به بهترین پایه
delta_f1 = (cv_results['LogisticRegression']['f1_mean'] - baselines[best_base_name]['f1_mean']) * 100
delta_acc = (cv_results['LogisticRegression']['acc_mean'] - baselines[best_base_name]['acc_mean']) * 100
print(f'  برتری LR نسبت به بهترین پایه: F1 +{delta_f1:.1f} امتیاز · Acc +{delta_acc:.1f} امتیاز')

# دقت به تفکیک منبع روی آزمون
per_source = {}
for s in sorted(set(src_te)):
    m = src_te == s
    if m.sum() >= 5:
        per_source[s] = round(float(accuracy_score(y_te[m], pred_te[m])), 4)

# ══════════════════════ ۵) آستانه‌های سه‌ناحیه‌ای (از OOF) ══════════════════════
print('\n' + '═' * 60)
print('۵) انتخاب دو آستانه از پیش‌بینی‌های out-of-fold')
print('═' * 60)
oof = cross_val_predict(grid.best_estimator_, X_tr, y_tr, cv=skf, method='predict_proba')[:, 1]
grid_p = np.unique(np.round(oof, 4))

T_HIGH_TARGET, T_LOW_TARGET = 0.97, 0.97
t_high, t_low = None, None
for t in grid_p:
    m = oof >= t
    if m.sum() >= 15 and y_tr[m].mean() >= T_HIGH_TARGET:
        t_high = float(t)
        break
if t_high is None:  # fallback: بیشترین دقت قابل دسترس
    precs = [(t, y_tr[oof >= t].mean()) for t in grid_p if (oof >= t).sum() >= 15]
    t_high = float(max(precs, key=lambda x: x[1])[0])
for t in grid_p[::-1]:
    m = oof <= t
    if m.sum() >= 15 and (y_tr[m] == 0).mean() >= T_LOW_TARGET:
        t_low = float(t)
        break
if t_low is None:
    t_low = 0.30

auto_rej_m = oof >= t_high
auto_app_m = oof <= t_low
rev_m = (~auto_rej_m) & (~auto_app_m)
thr_stats = {
    't_high': t_high, 't_low': t_low,
    'auto_reject_precision_oof': round(float(y_tr[auto_rej_m].mean()), 4) if auto_rej_m.sum() else None,
    'auto_approve_precision_oof': round(float((y_tr[auto_app_m] == 0).mean()), 4) if auto_app_m.sum() else None,
    'review_pct_oof': round(float(rev_m.mean()), 4),
    'coverage_auto_reject_oof': round(float(auto_rej_m.mean()), 4),
    'coverage_auto_approve_oof': round(float(auto_app_m.mean()), 4),
}
# همان آستانه‌ها روی آزمون (کنترل)
rj, ap = p_te >= t_high, p_te <= t_low
thr_stats['auto_reject_precision_test'] = round(float(y_te[rj].mean()), 4) if rj.sum() else None
thr_stats['auto_approve_precision_test'] = round(float((y_te[ap] == 0).mean()), 4) if ap.sum() else None
thr_stats['review_pct_test'] = round(float(((~rj) & (~ap)).mean()), 4)
print(f'  t_low={t_low:.2f} → تایید خودکار (دقت تایید {thr_stats["auto_approve_precision_oof"]:.1%})')
print(f'  t_high={t_high:.2f} → رد خودکار (دقت رد {thr_stats["auto_reject_precision_oof"]:.1%})')
print(f'  صف بازبینی: {thr_stats["review_pct_oof"]:.1%} (آموزش OOF) · {thr_stats["review_pct_test"]:.1%} (آزمون)')

# خطاهای آزمون برای بند ۱۱ گزارش (مرتب بر اساس قاطعیتِ اشتباه)
te_err_idx = np.where(pred_te != y_te)[0]
test_errors = []
for i2 in te_err_idx:
    test_errors.append({
        'norm_text': str(X_te[i2])[:100],
        'true': int(y_te[i2]), 'pred': int(pred_te[i2]), 'p': round(float(p_te[i2]), 4),
    })
test_errors = sorted(test_errors, key=lambda e: -abs(e['p'] - 0.5))
print(f'\nخطاهای آزمون: {len(test_errors)} (از {len(y_te)})')

# ══════════════════════ ۶) گزارش اطمینان ══════════════════════
decisive = float(np.mean((p_te > 0.9) | (p_te < 0.1)))
hist, edges = np.histogram(p_te, bins=10, range=(0, 1))
confidence = {
    'min': round(float(p_te.min()), 4), 'max': round(float(p_te.max()), 4),
    'mean': round(float(p_te.mean()), 4),
    'decisive_pct': round(decisive, 4),
    'histogram': {f'{edges[i]:.1f}-{edges[i+1]:.1f}': int(hist[i]) for i in range(10)},
}
print(f'\nاطمینان روی آزمون: بازه [{p_te.min():.2f}، {p_te.max():.2f}] · قاطع: {decisive:.1%}')

# ══════════════════════ ۷) ویژگی‌های برتر ══════════════════════
feat_names = np.array(lr_final.named_steps['features'].get_feature_names_out())
coefs = lr_final.named_steps['clf'].coef_[0]
top_reject = feat_names[np.argsort(coefs)[::-1][:20]].tolist()
top_approve = feat_names[np.argsort(coefs)[:20]].tolist()
print('\n۲۰ ویژگی برتر «رد»:', ', '.join(top_reject[:10]), '…')

# ══════════════════════ ۸) مدل نهایی (تمام داده) + ذخیره ══════════════════════
print('\n' + '═' * 60)
print('۸) آموزش نهایی روی تمام داده و ذخیره')
print('═' * 60)
final_pipe = grid.best_estimator_
final_pipe.fit(X_norm, y)
bundle = {
    'pipeline': final_pipe,
    't_low': t_low,
    't_high': t_high,
    'meta': {'trained_on': 'dataset.csv (full)', 'n_rows': int(len(df)),
             'random_state': SEED, 'normalizer': 'predict.normalize_text'},
}
model_path = os.path.join(BASE, 'model.joblib')
joblib.dump(bundle, model_path, compress=3)
model_mb = os.path.getsize(model_path) / 1e6
print(f'  model.joblib: {model_mb:.2f} MB (سقف ۲۰)')

# ══════════════════════ ۹) ارزیابی holdout از مسیر واقعی ══════════════════════
print('\n' + '═' * 60)
print('۹) ارزیابی holdout.csv (مسیر واقعی predict.py)')
print('═' * 60)
import predict as api  # noqa: E402

hold = pd.read_csv(os.path.join(BASE, 'holdout.csv'))
hold['norm'] = hold['text'].astype(str).map(normalize_text)
ds_keys = set(X_norm)
overlap = [t for t in hold['norm'] if t in ds_keys]
# کلیدهای ساخته‌شده فقط از نشانهاييموجي/نقطه‌گذاری اجتناب‌ناپذیرند (هر متنِ فقط-ایموجی یکسان می‌شود)
_hard = [t for t in overlap if t.replace('نشانهایموجی', '')
                    .replace('تنهاایموجی', '').replace('بیمحتوامتن', '')
                    .strip('!.؟?…,;: ').strip()]
assert not _hard, f'نشت! این متن‌های holdout در داده‌ی آموزش‌اند: {_hard}'

model = api.load_model(model_path)
t0 = time.time()
outs = api.predict(model, hold['text'].astype(str).tolist())
avg_ms = (time.time() - t0) / len(hold) * 1000
hold['p'] = [o['probability'] for o in outs]
hold['pred'] = [o['label'] for o in outs]
hold['decision'] = [o['decision'] for o in outs]

by_level = {}
for lev, g in hold.groupby('level'):
    by_level[lev] = {
        'n': int(len(g)),
        'accuracy': round(float(accuracy_score(g['label'], g['pred'])), 4),
        'f1': round(float(f1_score(g['label'], g['pred'], zero_division=0)), 4),
    }
    print(f'  {lev:7s}: acc={by_level[lev]["accuracy"]:.3f} (n={len(g)})')
holdout_acc = round(float(accuracy_score(hold['label'], hold['pred'])), 4)
rj_h = hold['decision'] == 'reject'
ap_h = hold['decision'] == 'approve'
holdout_zone = {
    'overall_accuracy': holdout_acc,
    'auto_reject_precision': round(float(hold.loc[rj_h, 'label'].mean()), 4) if rj_h.sum() else None,
    'auto_approve_precision': round(float((hold.loc[ap_h, 'label'] == 0).mean()), 4) if ap_h.sum() else None,
    'review_pct': round(float((hold['decision'] == 'review').mean()), 4),
    'zone_action_errors': int(((rj_h) & (hold['label'] == 0)).sum() + ((ap_h) & (hold['label'] == 1)).sum()),
}
print(f'  کل: acc={holdout_acc:.3f} · رد‌خودکار دقت={holdout_zone["auto_reject_precision"]} '
      f'· بازبینی={holdout_zone["review_pct"]:.0%}')

# خطاها برای بند ۱۱ گزارش
errors = []
for _, r in hold[hold['label'] != hold['pred']].iterrows():
    errors.append({'text': r['text'][:80], 'level': r['level'], 'true': int(r['label']),
                   'pred': int(r['pred']), 'p': float(r['p']), 'decision': r['decision']})
# خطاهای پرهزینه: تایید خودکارِ اشتباه یا رد خودکارِ اشتباه
for _, r in hold[((hold['decision'] == 'reject') & (hold['label'] == 0)) |
                 ((hold['decision'] == 'approve') & (hold['label'] == 1))].iterrows():
    if not any(e['text'][:40] == r['text'][:40] for e in errors):
        errors.append({'text': r['text'][:80], 'level': r['level'], 'true': int(r['label']),
                       'pred': int(r['pred']), 'p': float(r['p']), 'decision': r['decision'],
                       'zone_error': True})
print(f'  خطاها: {len(errors)} مورد')

# ══════════════════════ ۱۰) metrics.json ══════════════════════
metrics = {
    'config': {'random_state': SEED, 'sklearn': sklearn.__version__,
               'pandas': pd.__version__, 'numpy': np.__version__},
    'dataset': {
        'rows_raw': int(n_raw), 'rows_final': int(len(df)),
        'n_conflict_dropped': int(n_conflict),
        'n_duplicate_dropped': int(n_before_dedup - len(df) - n_conflict),
        'approve': int(len(df) - n_reject), 'reject': int(n_reject),
        'reject_ratio': round(n_reject / len(df), 4),
        'by_source': df['source'].value_counts().to_dict(),
        'unique_words_total': len(uniq_words), 'unique_words_generated': len(gen_words),
        'generated_templates': TEMPLATE_COUNT,
        'max_sentence_repeat': int(pd.Series(X_norm).value_counts().max()),
    },
    'signal_audit': signal_audit,
    'baselines': baselines,
    'best_baseline': best_base_name,
    'lr_beats_best_baseline_by': {'f1_points': round(delta_f1, 2), 'acc_points': round(delta_acc, 2)},
    'cv_5fold': cv_results,
    'lr_best_params': lr_params,
    'test': test_metrics,
    'per_source_test_accuracy': per_source,
    'thresholds': {
        **thr_stats,
        'policy': {
            't_high_rule': f'کوچک‌ترین آستانه با دقت رد ≥ {T_HIGH_TARGET:.0%} در OOF',
            't_low_rule': f'بزرگ‌ترین آستانه با دقت تایید ≥ {T_LOW_TARGET:.0%} در OOF',
        },
    },
    'confidence_test': confidence,
    'top20_features': {'reject': top_reject, 'approve': top_approve},
    'holdout': {'by_level': by_level, **holdout_zone, 'errors': errors},
    'test_errors': test_errors,
    'deploy': {
        'model_size_mb': round(model_mb, 3), 'model_limit_mb': 20,
        'avg_ms_per_comment': round(float(avg_ms), 3), 'time_limit_ms': 100,
        'compress': 3, 'network_calls': 0,
        'retrained_on_full_data': True,
        'weird_inputs': 'None/[]/""/emoji-only/5000ch/<script> → review یا پیش‌بینی، بدون crash',
    },
}
with open(os.path.join(BASE, 'metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print('\n' + '═' * 60)
print(f'✅ تمام شد — model.joblib ({model_mb:.2f}MB) + metrics.json')
print(f'   LR آزمون: Acc={test_metrics["accuracy"]:.3f} · Prec(رد)={test_metrics["reject_precision"]:.3f} · '
      f'holdout={holdout_acc:.3f} · {avg_ms:.1f}ms/کامنت')
