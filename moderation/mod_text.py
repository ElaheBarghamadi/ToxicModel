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


def normalize(text: str) -> str:
    t = str(text)
    t = _URL.sub(' لینک ', t)
    t = _MENTION.sub(' منشن ', t)
    t = t.translate(_CHAR_MAP)
    t = _DIACRITICS.sub('', t)
    t = _INVISIBLE.sub('', t)
    t = t.lower()
    t = _DIGITS.sub(' 0 ', t)
    t = _REPEATS.sub(r'\1\1', t)
    t = _WS.sub(' ', t).strip()
    return t


def dual_form(texts):
    """متن + نسخه‌ی بدون فاصله — برای مقابله با تله‌گذاری حروف مثل «ک ی ر».

    در شاخه‌ی char n-gram استفاده می‌شود تا الگوهای شکسته‌شده هم پیدا شوند.
    ورودی می‌تواند یک رشته یا لیستی از رشته‌ها باشد.
    """
    if isinstance(texts, str):
        texts = [texts]
    return [t + '\n' + t.replace(' ', '') for t in texts]
