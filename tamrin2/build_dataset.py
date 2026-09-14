# -*- coding: utf-8 -*-
"""
build_dataset.py — ساخت dataset.csv نهایی تمرین ۲ از سه منبع + داده‌ی تولیدی

منابع (به ترتیب اولویت در حذف تکراری):
  1. real_tutorials_labeled.csv  — ۸۱۰ کامنت واقعی یوتیوب (آموزش برنامه‌نویسی)، دست‌لیبل
  2. real_gold_all.csv (فقط توهین‌ها) — ۷۸ کامنت واقعی توهین‌آمیز (دسته‌ی ب)
  3. spammodel_curated.csv       — ۳۷۵ نمونه‌ی دستچین از ریپوی SpamModel، بازلیبل‌شده طبق قانون این تمرین
  4. generated.csv               — داده‌ی تولیدی برای دسته‌های خلوت + ضدمیان‌برها

نظافت (بند ۴.۳ صورت تمرین): dropna، حذف تکراری با کلید نرمال‌شده،
حذف کل گروهِ متن‌های دوجداری با برچسب متفاوت.
"""
import csv
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import normalize_text

BASE = os.path.dirname(os.path.abspath(__file__))


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


rows = []  # (text, label, source, origin_id)

# ۰) داده‌ی مشترک استاد (comments_seed.csv) — اگر موجود باشد به‌صورت خودکار اضافه می‌شود
seed_path = os.path.join(BASE, 'data', 'comments_seed.csv')
if os.path.exists(seed_path):
    n_seed = 0
    for r in load_csv(seed_path):
        rows.append((r['text'], int(r['label']), 'pycourse-seed', r.get('id', 'seed')))
        n_seed += 1
    print(f'comments_seed.csv استاد اضافه شد: {n_seed} ردیف (source=pycourse-seed)')
else:
    print('(comments_seed.csv استاد هنوز دریافت نشده — بدون آن ادامه می‌دهیم)')

# ۱) کامنت‌های واقعی یوتیوب (آموزش برنامه‌نویسی)
for r in load_csv(os.path.join(BASE, 'data', 'real_tutorials_labeled.csv')):
    rows.append((r['text'], int(r['label']), r['source'], r['id']))

# ۲) توهین‌های واقعی از gold قبلی (فقط label==1 → دسته‌ی ب این تمرین)
for r in load_csv(os.path.join(BASE, '..', 'data-website', 'real_gold_all.csv')):
    if r['label'] == '1':
        rows.append((r['text'], 1, 'youtube-real-insults', 'gold'))

# ۳) زیرمجموعه‌ی دستچین SpamModel
for r in load_csv(os.path.join(BASE, 'data', 'spammodel_curated.csv')):
    rows.append((r['text'], int(r['label']), r['source'], r['id']))

# ۴) داده‌ی تولیدی
for r in load_csv(os.path.join(BASE, 'data', 'generated.csv')):
    rows.append((r['text'], int(r['label']), r['source'], r['id']))

print(f'ورودی خام: {len(rows)} ردیف')

# ─── نظافت ───
clean = []
for text, label, source, oid in rows:
    if text is None or str(text).strip() == '':
        continue
    clean.append((str(text), label, source, oid))
n_after_na = len(clean)

# کلید نرمال‌شده (مرررسی/مرسی یکی می‌شوند، ف ا ل و/فالو یکی می‌شوند)
keys = [normalize_text(t) for t, *_ in clean]

# متن با دو برچسب متفاوت → حذف کل گروه
by_key = {}
for (text, label, source, oid), k in zip(clean, keys):
    by_key.setdefault(k, []).append((text, label, source, oid))
conflict_keys = {k for k, v in by_key.items() if len({x[1] for x in v}) > 1}
n_conflict = sum(len(by_key[k]) for k in conflict_keys)

# تکراری‌ها با کلید نرمال‌شده (نخستین occurrences — منابع واقعی جلوترند)
seen, final, n_dup = set(), [], 0
for (text, label, source, oid), k in zip(clean, keys):
    if k in seen or k in conflict_keys or k == '':
        n_dup += 1
        continue
    seen.add(k)
    final.append((text, label, source, oid))

print(f'dropna: {len(rows)-n_after_na} · متن با برچسب متناقض (کل گروه): {n_conflict} · '
      f'تکراری/خالی‌شده: {n_dup} → نهایی: {len(final)}')

# ─── QA: تاییدهای مشکوک به بی‌محتوا (باید دست‌برچسب شوند) ───
suspicious = [(t, s) for (t, l, s, o), k in zip(final, [normalize_text(t) for t, *_ in final])
              if l == 0 and k and not any(ch.isalpha() for ch in k)]
if suspicious:
    print('⚠️ تاییدهای بدون حرف (بررسی دستی لازم):')
    for t, s in suspicious[:10]:
        print('   ', repr(t[:60]), s)

# ─── خروجی ───
n_rej = sum(l for _, l, _, _ in final)
uniq_words = set()
gen_words = set()
for (t, l, s, o) in final:
    k = normalize_text(t)
    toks = k.split()
    uniq_words.update(toks)
    if s.startswith('gen-'):
        gen_words.update(t.split())

with open(os.path.join(BASE, 'dataset.csv'), 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'text', 'label', 'source'])
    for i, (t, l, s, o) in enumerate(final, 1):
        w.writerow([f't2-{i:04d}', t, l, s])

with open(os.path.join(BASE, 'data', 'id_map.csv'), 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t2_id', 'origin_id', 'source'])
    for i, (t, l, s, o) in enumerate(final, 1):
        w.writerow([f't2-{i:04d}', o, s])

src = Counter(s for _, _, s, _ in final)
print(f'\n✅ dataset.csv: {len(final)} ردیف · {n_rej} رد ({n_rej/len(final):.1%}) · {len(final)-n_rej} تایید')
print(f'کلمات یکتا (کل، پس از نرمال‌سازی): {len(uniq_words)} · کلمات یکتای بخش تولیدی: {len(gen_words)}')
print('به تفکیک منبع:')
for s, c in src.most_common():
    n_r = sum(1 for _, l, ss, _ in final if ss == s and l == 1)
    print(f'   {s:28s} {c:4d} ({n_r} رد)')
