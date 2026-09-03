# -*- coding: utf-8 -*-
"""پاکسازی دیتاست persian-abusive-words (هابینگ‌فیس):

- ادغام train.csv و test.csv رسمی
- نرمال‌سازی یونیکد + حذف تکراری‌ها و برچسب‌های متناقض
- تقسیم جدید طبقه‌بندی‌شده ۸۵/۱۵ (seed=42) با صفر هم‌پوشانی

ورودی:  persian-abusive-words/train.csv و test.csv  (دانلود از:
        https://huggingface.co/datasets/AlirezaFzp/persian-abusive-words)
خروجی: persian-abusive-words/clean/train_clean.csv و test_clean.csv
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mod_text import normalize

ROOT = Path(__file__).resolve().parent.parent / 'persian-abusive-words'
OUT = ROOT / 'clean'


def load(p):
    with open(p, encoding='utf-8-sig', errors='replace') as f:
        r = csv.reader(f)
        next(r)
        return [(fl[0], fl[1].strip()) for fl in r
                if len(fl) == 2 and fl[1].strip() in ('0', '1')]


def main():
    from sklearn.model_selection import train_test_split
    raw = load(ROOT / 'train.csv') + load(ROOT / 'test.csv')
    print('raw:', len(raw), dict(Counter(l for _, l in raw)))

    lab = defaultdict(set)
    for t, l in raw:
        lab[normalize(t)].add(l)
    pairs = [(t, int(list(s)[0])) for t, s in lab.items() if len(s) == 1 and t]
    print(f'unique: {len(lab)} | conflicting dropped: {sum(1 for s in lab.values() if len(s) > 1)}')
    print('kept:', len(pairs), dict(Counter(l for _, l in pairs)))

    X = [t for t, _ in pairs]
    y = [l for _, l in pairs]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    assert not set(Xtr) & set(Xte)

    OUT.mkdir(exist_ok=True)
    for name, xs, ys in [('train_clean.csv', Xtr, ytr), ('test_clean.csv', Xte, yte)]:
        with open(OUT / name, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['tweet', 'label'])
            w.writerows(zip(xs, ys))
        print(name, len(xs), dict(Counter(ys)))


if __name__ == '__main__':
    main()
