# -*- coding: utf-8 -*-
"""
predict.py — رابط پیش‌بینیِ تمرین ۲ (تعدیل کامنت pycourse.ir)

قرارداد (دقیقاً مطابق صورت تمرین):

    model  = load_model("model.joblib")
    outs   = predict(model, ["متن کامنت ۱", "متن کامنت ۲"])
    # هر خروجی: {"probability": float, "decision": "approve/review/reject", "label": 0/1}

محدودیت‌های اجرا: بدون شبکه، بدون وابستگی به جز scikit-learn/pandas/numpy/joblib/scipy،
مقاوم به ورودی‌های عجیب (رشته‌ی خالی، فقط ایموجی، متن ۵۰۰۰ کاراکتری، None، لیست خالی، تگ script).
اگر بعد از پاکسازی چیزی از متن نماند → تصمیم «review» است، نه یک حدس تصادفی.
"""

import re
import sys
import json

import joblib

# ─────────────────────────── نرمال‌سازی فارسی ───────────────────────────
# همین تابع در train.py (با import از همین فایل) برای ساخت داده‌ی آموزش استفاده می‌شود؛
# یعنی آموزش و استنتاج دقیقاً یک پیش‌پردازش می‌بینند.

# ارقام فارسی/عربی → لاتین
_DIGITS = {}
for _i, _d in enumerate('۰۱۲۳۴۵۶۷۸۹'):
    _DIGITS[_d] = str(_i)
for _i, _d in enumerate('٠١٢٣٤٥٦٧٨٩'):
    _DIGITS[_d] = str(_i)

# نشانه‌ها به‌جای حذف، به توکن تبدیل می‌شوند (تله ۵ صورت تمرین: سیگنال را نگه دار، حذف نکن)
_URL_RE = re.compile(r'(?:https?://|www\.)\S+')
_DOMAIN_RE = re.compile(
    r'\b(?:[\w\-]+\.)+(?:com|ir|org|net|io|me|xyz|site|dev|app|ai|co|info|shop|online|tv|link)(?:/\S*)?',
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r'\b[\w\.\-]+@[\w\.\-]+\.\w{2,}\b')
# شماره موبایل: ارقام جدا یا چسبیده (۰۹۱۲… ، ۰ ۹ ۱ ۲ … ، 9123456789)
_PHONE_RE = re.compile(r'(?:\+?98|0098|0)?\s*9(?:\s?\d){9}\b')
# شماره‌ی نوشته‌شده با حروف: ۷+ کلمه‌ی رقم‌پشت‌سرهم («صفر نه یک دو …»)
_DW = r'(?:صفر|یک|یه|دو|دوه|سه|چار|چهار|پنج|شش|شیش|هفت|هشت|نه)'
_PHONE_WORDS_RE = re.compile(_DW + r'(?:\s+' + _DW + r'){6,}')
# حروف تکراری کشیده: «سلاااام» → «سلام» (فقط حروف؛ ارقام دست‌نخورده می‌مانند)
_STRETCH_RE = re.compile(r'([^\W\d])\1{2,}', re.UNICODE)
# ایموجی (نجوا/چهره/نمادها)
_EMOJI_RE = re.compile(
    '['
    '\U0001F000-\U0001FAFF'   # emoji اصلی
    '\u2600-\u27BF'            # نمادها و dingbats
    '\u2B00-\u2BFF'            # فلش‌ها و نمادهای دیگر
    '\u2190-\u21FF'            # فلش‌ها
    '\uFE0F'                   # variation selector
    ']+'
)


def _join_single_char_runs(toks):
    """«ف ا ل و کنید» → «فالو کنید» — خنثی‌سازی فاصله‌گذاریِ عمدی برای گریز از فیلتر.
    فقط اجرِ ۳+ توکن تک‌حرفی چسبانده می‌شود؛ «سه و چهار» (تک‌حرفیِ پراکنده) دست‌نخورده می‌ماند."""
    out, run = [], []
    for t in toks:
        if len(t) == 1:
            run.append(t)
        else:
            if len(run) >= 3:
                out.append(''.join(run))
            elif run:
                out.extend(run)
            run = []
            out.append(t)
    if len(run) >= 3:
        out.append(''.join(run))
    elif run:
        out.extend(run)
    return out



_MARKERS = {'نشانهلینک', 'نشانهتلفن', 'نشانهایمیل'}
_WORDISH = re.compile(r'[A-Za-z\u0600-\u06FF]{2}')


def _mark_contentless(toks):
    """قانون د (بی‌محتوا) به‌صورت توکن: اگر متن هیچ واژه‌ی واقعی نداشته باشد،
    نشانه‌ی «تنهاایموجی» یا «بیمحتوامتن» به آن اضافه می‌شود تا مدل سیگنال را ببیند.
    (رشته‌ی کاملاً خالی عمداً خالی می‌ماند → تصمیم «صف بازبینی»)"""
    has_real = any(t != 'نشانهایموجی' and (_WORDISH.search(t) or t in _MARKERS) for t in toks)
    if toks and not has_real:
        if any(t == 'نشانهایموجی' for t in toks):
            toks = toks + ['تنهاایموجی']
        else:
            toks = toks + ['بیمحتوامتن']
    return toks


def normalize_text(text):
    """نرمال‌سازی کامل: یکسان‌سازی ي/ی و ك/ک، ارقام فارسی/عربی → لاتین، نیم‌فاصله،
    حروف کشیده، لینک/ایمیل/تلفن/ایموجی → توکن نشانه، و چسباندن اجرِ تک‌حرفی‌ها."""
    if text is None:
        return ''
    s = str(text)
    if len(s) > 3000:          # متن خیلی طولانی → برش برای حفظ سرعت
        s = s[:3000]
    # تگ‌های HTML به‌همراه محتوایشان (مثل <script>alert(1)</script>) حذف می‌شوند
    s = re.sub(r'&[a-zA-Z#0-9]+;', ' ', s)
    s = re.sub(r'<script\b.*?</script\s*>', ' ', s, flags=re.IGNORECASE | re.DOTALL)
    s = re.sub(r'<[^>]*>', ' ', s)
    for a, b in (('ي', 'ی'), ('ك', 'ک'), ('ة', 'ه'), ('أ', 'ا'), ('إ', 'ا'), ('ؤ', 'و'), ('ئ', 'ی')):
        s = s.replace(a, b)
    for d, e in _DIGITS.items():
        s = s.replace(d, e)
    # نیم‌فاصله و علائم کنترل جهت‌دهی حذف (می‌خوام → میخوام)
    for ch in ('\u200c', '\u200d', '\u200e', '\u200f', '\ufeff', '\u0640'):
        s = s.replace(ch, '')
    s = _STRETCH_RE.sub(r'\1', s)
    s = _URL_RE.sub(' نشانهلینک ', s)
    s = _EMAIL_RE.sub(' نشانهایمیل ', s)
    s = _DOMAIN_RE.sub(' نشانهلینک ', s)
    s = _PHONE_RE.sub(' نشانهتلفن ', s)
    s = _PHONE_WORDS_RE.sub(' نشانهتلفن ', s)
    s = _EMOJI_RE.sub(' نشانهایموجی ', s)
    toks = _join_single_char_runs(s.split())
    toks = _mark_contentless(toks)
    return ' '.join(toks).strip()


# ─────────────────────────── قرارداد پیش‌بینی ───────────────────────────

def load_model(path: str = "model.joblib"):
    """مدل را بارگذاری می‌کند و برمی‌گرداند (pipeline + دو آستانه‌ی سه‌ناحیه‌ای)."""
    return joblib.load(path)


def predict(model, texts: list):
    """
    برای هر متن یک دیکشنری با سه کلید برمی‌گرداند:
      {"probability": float,   # احتمال مشکل‌دار بودن (0..1)
       "decision": str,        # approve / review / reject
       "label": int}           # 0 یا 1 (بر اساس p >= 0.5)
    """
    if texts is None:
        return []
    if isinstance(texts, str):          # اگر یک رشته‌ی تنها داده شد
        texts = [texts]
    texts = list(texts)

    if isinstance(model, dict) and 'pipeline' in model:
        pipe = model['pipeline']
        t_low = float(model.get('t_low', 0.35))
        t_high = float(model.get('t_high', 0.65))
    else:                                # سازگاری: pipeline خام بدون آستانه
        pipe = model
        t_low, t_high = 0.35, 0.65

    normed = [normalize_text(t) for t in texts]
    probs = [None] * len(normed)
    idx = [i for i, s in enumerate(normed) if s]
    if idx:
        p_vec = pipe.predict_proba([normed[i] for i in idx])[:, 1]
        for i, p in zip(idx, p_vec):
            probs[i] = float(p)

    results = []
    for s, p in zip(normed, probs):
        if p is None:
            # بعد از پاکسازی چیزی نمانده → صف بازبینی (نه حدس تصادفی)
            p = float(pipe.predict_proba([''])[0, 1])
            decision = 'review'
        elif p >= t_high:
            decision = 'reject'
        elif p <= t_low:
            decision = 'approve'
        else:
            decision = 'review'
        results.append({
            'probability': round(p, 4),
            'decision': decision,
            'label': int(p >= 0.5),
        })
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='پیش‌بینی برچسب کامنت‌ها')
    ap.add_argument('texts', nargs='*', help='متن(های) کامنت')
    ap.add_argument('--file', help='فایل CSV با ستون text')
    ap.add_argument('--model', default='model.joblib')
    args = ap.parse_args()
    if args.file:
        import pandas as pd
        texts = pd.read_csv(args.file)['text'].astype(str).tolist()
    elif args.texts:
        texts = args.texts
    else:                                 # نمونه‌ی خودآزمایی
        texts = ['ممنون استاد عالی بود', 'ف ا ل و کنید کانال ما رو', '']
    m = load_model(args.model)
    print(json.dumps(predict(m, texts), ensure_ascii=False, indent=2))
