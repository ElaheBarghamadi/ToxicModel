# -*- coding: utf-8 -*-
"""پردازش فایل‌های آپلودی (CSV/XLSX) + محاسبه متریک‌های ارزیابی."""
import csv
import io
import re
from pathlib import Path

TEXT_COLS = ['text', 'tweet', 'comment', 'متن', 'کامنت', 'نظر', 'توییت']
LABEL_COLS = ['label', 'class', 'target', 'لیبل', 'برچسب']

FA_LABELS = {
    '1': 1, '0': 0, '1.0': 1, '0.0': 0,
    'offensive': 1, 'neutral': 0, 'normal': 0, 'abusive': 1, 'clean': 0,
    'yes': 1, 'no': 0, 'true': 1, 'false': 0,
}


def parse_file(fileobj):
    """فایل CSV یا XLSX را می‌خواند → (ردیف‌ها, نام ستون متن, نام ستون لیبل|None)."""
    name = (fileobj.name or '').lower()
    raw = fileobj.read()
    if name.endswith(('.xlsx', '.xls')):
        import pandas as pd
        df = pd.read_excel(io.BytesIO(raw))
        cols = [str(c).strip().lower() for c in df.columns]
        text_col = next((df.columns[i] for i, c in enumerate(cols) if c in TEXT_COLS), df.columns[0])
        label_col = next((df.columns[i] for i, c in enumerate(cols) if c in LABEL_COLS), None)
        rows = []
        for _, r in df.iterrows():
            t = str(r[text_col]).strip()
            lab = None
            if label_col is not None and str(r[label_col]).strip().lower() in FA_LABELS:
                lab = FA_LABELS[str(r[label_col]).strip().lower()]
            if t and t.lower() != 'nan':
                rows.append((t, lab))
        return rows, str(text_col), str(label_col) if label_col else None

    # CSV / TSV — با بررسی جداکننده
    text = raw.decode('utf-8-sig', errors='replace')
    delim = '\t' if '\t' in text.splitlines()[0] else ','
    f = io.StringIO(text)
    reader = csv.reader(f, delimiter=delim)
    try:
        header = [h.strip().lower() for h in next(reader)]
    except StopIteration:
        return [], None, None
    ti = next((i for i, c in enumerate(header) if c in TEXT_COLS), 0)
    li = next((i for i, c in enumerate(header) if c in LABEL_COLS), None)
    rows = []
    for fl in reader:
        if len(fl) <= max(ti, li or 0):
            continue
        t = fl[ti].strip()
        if not t:
            continue
        lab = None
        if li is not None:
            v = fl[li].strip().lower()
            if v in FA_LABELS:
                lab = FA_LABELS[v]
        rows.append((t, lab))
    return rows, header[ti] if header else None, header[li] if li is not None else None


def evaluate(rows_with_labels):
    """محاسبه متریک‌ها؛ «پیش‌بینی مثبت» یعنی تصمیم review یا block."""
    tp = fp = fn = tn = 0
    decisions = {'ok': 0, 'review': 0, 'block': 0}
    for _, label, decision in rows_with_labels:
        decisions[decision] += 1
        pred = 1 if decision in ('review', 'block') else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1:
            fp += 1
        elif label == 1:
            fn += 1
        else:
            tn += 1
    P = tp / (tp + fp) if tp + fp else None
    R = tp / (tp + fn) if tp + fn else None
    F1 = (2 * P * R / (P + R)) if P and R else None
    acc = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': P, 'recall': R, 'f1': F1, 'accuracy': acc,
            'decisions': decisions, 'n': len(rows_with_labels)}
