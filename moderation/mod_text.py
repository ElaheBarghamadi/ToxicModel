# -*- coding: utf-8 -*-
"""نرمال‌سازی متن فارسی — مشترک بین آموزش و استنتاج.

این ماژول باید کنار model.joblib بماند چون پایپ‌لاین مدل به dual_form آن ارجاع دارد.
"""
import re

# یکسان‌سازی حروف عربی/فارسی
_CHAR_MAP = str.maketrans({
    'ي': 'ی', 'ك': 'ک', 'إ': 'ا', 'أ': 'ا', 'ٱ': 'ا', 'ؤ': 'و',
    'ۓ': 'ی', 'ة': 'ه', 'ۀ': 'ه', 'ڑ': 'ر', 'ڕ': 'ر',
})

# اعراب و کشیده و نویسه‌های نامرئی (نیم‌فاصله، نشانه‌های جهت و ...)
_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u0640]')
_INVISIBLE = re.compile(r'[\u200b\u200c\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]')

_URL = re.compile(r'(?:https?://|www\.)\S+', re.I)
_MENTION = re.compile(r'@[A-Za-z0-9_]+')
_DIGITS = re.compile(r'[0-9۰-۹٠-٩]+')
_REPEATS = re.compile(r'(.)\1{2,}', re.S)   # «بخداااااا» → «بخداا»
_WS = re.compile(r'\s+')

# نقطه/خط‌اتصال بین حروف فارسی: «ک.ی.ر» یا «ک_ی_ر» → «کیر»
_GLUE_IN_WORD = re.compile(r'(?<=[\u0600-\u06FF])[.\-_ـ\u200c]+(?=[\u0600-\u06FF])')

# ── فینگلیش → فارسی (برای کشف فحش لاتین‌نویسی: «kire to in gheymat») ──
_DIGRAPHS = [('sh', 'ش'), ('ch', 'چ'), ('zh', 'ژ'), ('kh', 'خ'), ('gh', 'غ'),
             ('ph', 'ف'), ('th', 'ث'), ('aa', 'آ'), ('ee', 'ی'), ('oo', 'و')]
_SINGLES = {'a': 'ا', 'b': 'ب', 'c': 'ک', 'd': 'د', 'e': 'ه', 'f': 'ف', 'g': 'گ',
            'h': 'ه', 'i': 'ی', 'j': 'ج', 'k': 'ک', 'l': 'ل', 'm': 'م', 'n': 'ن',
            'o': 'و', 'p': 'پ', 'q': 'ق', 'r': 'ر', 's': 'س', 't': 'ت', 'u': 'و',
            'v': 'و', 'w': 'و', 'x': 'خ', 'y': 'ی', 'z': 'ز'}
_LATIN_TOKEN = re.compile(r'[a-z]{2,}')


def _translit(token: str) -> str:
    out, i, n = [], 0, len(token)
    while i < n:
        two = token[i:i + 2]
        if two in dict(_DIGRAPHS):
            out.append(dict(_DIGRAPHS)[two]); i += 2; continue
        out.append(_SINGLES.get(token[i], '')); i += 1
    return ''.join(out)


def normalize(text: str) -> str:
    t = str(text)
    t = _URL.sub(' لینک ', t)
    t = _MENTION.sub(' منشن ', t)
    t = t.translate(_CHAR_MAP)
    t = _DIACRITICS.sub('', t)
    t = _INVISIBLE.sub('', t)
    t = _GLUE_IN_WORD.sub('', t)          # «ک.ی.ر» → «کیر»
    t = t.lower()
    t = _DIGITS.sub(' 0 ', t)
    t = _REPEATS.sub(r'\1\1', t)
    t = _WS.sub(' ', t).strip()
    # فینگلیش: معادلِ حرف‌نویسی لاتین را به انتهای متن می‌چسبانیم
    tr = _LATIN_TOKEN.sub(lambda m: _translit(m.group(0)), t)
    if tr != t:
        t = t + ' ' + tr
    return t


def dual_form(texts):
    """متن + نسخه‌ی بدون فاصله — برای مقابله با تله‌گذاری حروف مثل «ک ی ر».

    در شاخه‌ی char n-gram استفاده می‌شود تا الگوهای شکسته‌شده هم پیدا شوند.
    ورودی می‌تواند یک رشته یا لیستی از رشته‌ها باشد.
    """
    if isinstance(texts, str):
        texts = [texts]
    return [t + '\n' + t.replace(' ', '') for t in texts]
