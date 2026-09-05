# -*- coding: utf-8 -*-
"""ادغام دیتاست فارسی-دری (Kaggle/yasinhazara، CC0) با بودجه ثابت ۱۹٬۸۰۰:

  ۱) تفکیک: ۲٬۵۰۰ نمونه تست دری (test_farsi_dari.csv — جدید) / بقیه استخر آموزش
  ۲) حذف هم‌پوشانی با تست‌های موجود (۵۲ مورد)
  ۳) جایگزینی ۱٬۸۰۰ نمونه easy غیر-توییتی از train_selected با ۱٬۸۰۰ نمونه دری
     (منبع = farsi_dari، برچسب reason = easy)
ورودی: candidates/farsi_dari_cleaned.csv ·  خروجی: train_selected.csv (بازنویسی) + test_farsi_dari.csv
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mod_text import normalize

HERE = Path(__file__).resolve().parent
MERGED = HERE.parent / 'data-merged'
DARI = HERE.parent / 'candidates' / 'farsi_dari_cleaned.csv'
SEED = 42
random.seed(SEED)

# ---------- ۱) خواندن دری + نرمال‌سازی + dedup با تست‌های موجود ----------
dari = []
with open(DARI, encoding='utf-8-sig') as f:
    r = csv.reader(f); next(r)
    for fl in r:
        if len(fl) >= 2 and fl[1].strip() in ('0', '1'):
            t = normalize(fl[0])[:250]
            if len(t) >= 4:
                dari.append((t, int(fl[1])))
seen = set()
dari = [(t, l) for t, l in dari if not (t in seen or seen.add(t))]

test_texts = set()
for p in MERGED.glob('test_*.csv'):
    with open(p, encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        test_texts.update(fl[0].strip() for fl in r if len(fl) >= 2)
dari = [(t, l) for t, l in dari if t not in test_texts]
print(f'دری پس از dedup: {len(dari)} (مثبت {sum(l for _, l in dari)})')

# ---------- ۲) تست دری: ۲٬۵۰۰ طبقه‌بندی‌شده ----------
by_lab = defaultdict(list)
for t, l in dari:
    by_lab[l].append((t, l))
test_dari, pool = [], []
for l, items in by_lab.items():
    random.shuffle(items)
    k = int(2500 * len(items) / len(dari))
    test_dari += items[:k]
    pool += items[k:]
random.shuffle(test_dari); random.shuffle(pool)

# ---------- ۳) جایگزینی در train_selected ----------
sel = []
with open(MERGED / 'train_selected.csv', encoding='utf-8') as f:
    r = csv.reader(f); next(r)
    sel = [fl for fl in r if len(fl) >= 4]
pool_texts = {t for t, _ in pool}
sel = [fl for fl in sel if fl[0] not in pool_texts]

# حذف ۱٬۸۰۰ easy غیر-توییتی (توییتر برای v1 نگه داشته می‌شود)
easy_nontw = [i for i, fl in enumerate(sel) if fl[3] == 'easy' and fl[2] != 'tweets']
random.shuffle(easy_nontw)
drop = set(easy_nontw[:1800])
sel = [fl for i, fl in enumerate(sel) if i not in drop]

by_cs = defaultdict(list)
for t, l in pool:
    by_cs[l].append((t, l))
add = []
for l, items in by_cs.items():
    k = int(1800 * len(items) / len(pool))
    add += [[t, str(l), 'farsi_dari', 'easy'] for t, l in items[:k]]
sel += add
random.shuffle(sel)

with open(MERGED / 'train_selected.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['text', 'label', 'source', 'reason'])
    w.writerows(sel)
with open(MERGED / 'test_farsi_dari.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['text', 'label', 'source'])
    w.writerows((t, l, 'farsi_dari') for t, l in test_dari)

print(f'train_selected: {len(sel)} | dari در آموزش: {sum(1 for fl in sel if fl[2]=="farsi_dari")}')
print('منابع:', dict(Counter(fl[2] for fl in sel)))
print(f'test_farsi_dari: {len(test_dari)} (مثبت {sum(l for _, l in test_dari)})')
