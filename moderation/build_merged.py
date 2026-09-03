# -*- coding: utf-8 -*-
"""ساخت مجموعه داده ادغام‌شده از ۵ منبع فارسی:

1. persian-abusive-words (توییتر، Apache-2.0)      — ستون فقرات فعلی
2. naseza (تلگرام، CC0)                              — توهین/نفرت تلگرامی
3. pars-offensive (کامنت اینستاگرام، مقاله‌ای)        — دقیقاً دامنه کامنت
4. phate (توییتر، AAAI24)                            — چندبرچسبی خشونت/فحش/نفرت
5. phicad (کامنت اینستاگرام، ۳۰۱ هزار)               — توهین ملایم + رکیک (زیرنمونه)

خروجی: data-merged/train_merged.csv و test ها به تفکیک منبع
"""
import csv, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, '/home/user/moderation')
from mod_text import normalize

random.seed(42)
ROOT = Path('/home/user')
OUT = ROOT / 'data-merged'
OUT.mkdir(exist_ok=True)

MAXLEN = 250

def add(store, source, text, label):
    t = normalize(text)[:MAXLEN]
    if len(t) >= 2:
        store.append((t, int(label), source))

# ---------- 1) دیتاست فعلی (تقسیم تمیز قبلی حفظ می‌شود) ----------
cur_train, cur_test = [], []
for path, store in [('persian-abusive-words/clean/train_clean.csv', cur_train),
                    ('persian-abusive-words/clean/test_clean.csv', cur_test)]:
    with open(ROOT / path, encoding='utf-8-sig') as f:
        r = csv.reader(f); next(r)
        for fl in r:
            if len(fl) == 2:
                add(store, 'tweets', fl[0], fl[1])

# ---------- 2) naseza (تلگرام) ----------
nas = json.load(open(ROOT / 'candidates/naseza/naseza.json', encoding='utf-8'))
nas_rows = []
for x in nas:
    labs = x.get('label') or []
    if not labs:
        continue
    lab = 1 if 'Offensive' in labs else 0        # («Normal» یا ترکیبی نادر حذف/صفر)
    if labs == ['Normal', 'Offensive']:
        continue
    add(nas_rows, 'naseza', x['text'], lab)

# ---------- 3) pars-offensive (اینستاگرام) ----------
po = pd.read_excel(ROOT / 'candidates/pars-offensive/ParsOffensive.xlsx')
po_rows = []
for _, row in po.iterrows():
    lab = 1 if str(row['label']).strip().lower() == 'offensive' else 0
    add(po_rows, 'pars_offensive', str(row['comment']), lab)

# ---------- 4) phate (توییتر) ----------
ph_rows = []
for part in ['train_simple', 'val_simple', 'test_simple']:
    df = pd.read_csv(ROOT / f'candidates/phate/{part}.csv')
    for _, row in df.iterrows():
        lab = int(row['Violenc'] or row['Hate'] or row['Vulgar'] or row['HateSpeech'])
        add(ph_rows, 'phate', str(row['text']), lab)

# ---------- 5) phicad (اینستاگرام، زیرنمونه متوازن) ----------
phicad_rows = []
by_class = defaultdict(list)
for part in ['PHICAD-part1.csv', 'PHICAD-part2.csv']:
    with open(ROOT / 'candidates/phicad' / part, encoding='utf-8', errors='replace') as f:
        r = csv.reader(f, delimiter='\t'); next(r)
        for fl in r:
            if len(fl) == 5 and fl[4]:
                by_class[fl[4]].append((fl[0], fl[4]))
POS_CLASSES = {'hate', 'obscene', 'hateobscene', 'spamobscene'}
CAPS = {'hate': 25000, 'clean': 25000, 'spam': 5000}   # obscene* کامل می‌آیند
for cl, items in by_class.items():
    random.shuffle(items)
    cap = CAPS.get(cl, len(items))
    items = items[:cap]
    for text, _ in items:
        add(phicad_rows, 'phicad', text, 1 if cl in POS_CLASSES else 0)

# ---------- تفکیک آموزش/تست هر منبع جدید (۱۵٪ تست) ----------
def split(rows):
    random.shuffle(rows)
    k = int(len(rows) * 0.15)
    return rows[k:], rows[:k]

train, tests = cur_train[:], {'tweets': cur_test}
for name, rows in [('naseza', nas_rows), ('pars_offensive', po_rows),
                   ('phate', ph_rows), ('phicad', phicad_rows)]:
    tr, te = split(rows)
    train += tr
    tests[name] = te

# ---------- حذف تکرار بین‌منبعی از آموزش (تست‌ها دست‌نخورده) ----------
test_texts = set()
for v in tests.values():
    test_texts.update(t for t, _, _ in v)
seen, deduped = set(), []
for t, l, s in train:
    if t in test_texts or t in seen:
        continue
    seen.add(t); deduped.append((t, l, s))
train = deduped

random.shuffle(train)

# ---------- ذخیره ----------
with open(OUT / 'train_merged.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f); w.writerow(['text', 'label', 'source'])
    w.writerows(train)

for name, rows in tests.items():
    with open(OUT / f'test_{name}.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['text', 'label', 'source'])
        w.writerows(rows)

print(f"TRAIN: {len(train)} | {dict(Counter(s for _,_,s in train))}")
print(f"  labels: {dict(Counter(l for _,l,_ in train))} ({Counter(l for _,l,_ in train)[1]/len(train):.1%} positive)")
for name, rows in tests.items():
    c = Counter(l for _, l, _ in rows)
    print(f"TEST {name:15s}: {len(rows):6d} | pos={c[1]} ({c[1]/len(rows):.1%})")

all_train_texts = set(t for t, _, _ in train)
for name, rows in tests.items():
    ov = len(all_train_texts & set(t for t, _, _ in rows))
    print(f"leak check {name}: {ov}")
