# 🛡️ ToxicModel — تشخیص کامنت نامناسب فارسی (v4)

سیستم تشخیص محتوای نامناسب (فحش و توهین) در کامنت‌های فارسی + رابط وب جنگو.

**English TL;DR:** Persian toxic-comment detection — single **Logistic Regression**,
Iranian-only data (5 sources), value-aware selection (19.8k), validation-derived
threshold, binary decision (ok/block), bootstrap CIs, rollback-safe retraining,
plus a Django app for single/batch testing (auto-scoring when labels exist).

> **الزام تکلیف:** الگوریتم مدل = **رگرسیون لجستیک (Logistic Regression)** — تک‌مدل.
> معماری ساده: فقط «مجاز / بلاک» — بدون صف بازبینی.

---

## نتایج نسخه ۴ (شروع تازه: فقط داده ایرانی، تک‌مدل، بدون بازبینی)

**داده آموزش: ۱۹٬۸۰۰ نمونه گزینش‌شده از ۵ منبع ایرانی** (بدون فارسی-دری)

| مجموعه آزمون | دامنه | F1 | دقت | بازیابی |
|---|---|---|---|---|
| توییتر | Persian Abusive Words | 0.733 | 0.672 | 0.806 |
| Naseza | تلگرام | 0.814 | 0.791 | 0.838 |
| ParsOffensive | کامنت اینستاگرام | 0.735 | 0.643 | 0.858 |
| PHate | توییتر | 0.702 | 0.682 | 0.724 |
| PHICAD | کامنت اینستاگرام | 0.879 | 0.925 | 0.837 |
| کامنت سایت (تولادی ۸۴تایی) | ارزیابی مستقل | 0.605 | 0.821 | 0.479 |
| **همه با هم (~۱۹٬۶۰۰)** | — | **0.847** | — | — |

### معماری تصمیم

```
p = LogisticRegression (C=4) · احتمال نامناسب‌بودن

p ≥ 0.509  →  🚫 block (نامناسب)
در غیر این صورت → ✅ ok (مجاز)
```

آستانه‌ها در `model_config.json` ذخیره‌اند و از آنجا خوانده می‌شوند.

## ایمن‌سازی خط آموزش (جدید v3)

1. **پالایش نشت:** داده آپلودی کاربر در برابر همه فایل‌های تست dedup می‌شود
2. **نسخه‌بندی:** قبل از بازنویسی، مدل در `models_archive/` بایگانی می‌شود (۳ نسخه) + دکمه بازگردانی در سایت
3. **سنجش سلامت:** اگر F1 تست توییتی < 0.45 بیامد، آموزش رد می‌شود و مدل قبلی حفظ می‌شود
4. **آستانه‌های صادقانه:** انتخاب آستانه فقط روی validation؛ تست‌ها یک‌بار خوانده می‌شوند + CI بوت‌استرپ

## ساختار ریپو

```
moderation/
├── model.joblib                     ← مدل فعال (LogisticRegression)
├── model_config.json                ← آستانه‌ها + نسخه + متادیتا
├── moderate.py                      ← API استنتاج (ok/review/block)
├── mod_text.py                      ← نرمال‌سازی فارسی + ترفند ضد فحش حرف‌جدا
├── clean_abusive_words.py           ← پاکسازی دیتاست پایه (dedupe + رفع نشت)
├── build_merged.py                  ← ساخت دیتاست ادغام ۵ منبع
├── train_model.py / _v2.py / _v3.py ← پایپ‌لاین‌های آموزش (v3 = نسخه نهایی)
├── train_web.py                     ← آموزش از وب (ایمن‌سازی‌شده)
└── metrics.json                     ← همه ارزیابی‌ها (v1/v2/v3)

modsite/                             ← سایت جنگو (RTL فارسی)
```

## اجرای سایت

```bash
pip install -r modsite/requirements.txt
cd modsite && python manage.py runserver     # http://127.0.0.1:8000
```

| صفحه | کار |
|---|---|
| `/` | داشبورد: نتایج + CI، توضیح متریک‌ها، قاعده تصمیم |
| `/test/` | تست تک‌کامنت (هر خط = یک کامنت) |
| `/batch/` | تست فایل CSV/XLSX — با ستون `label` → Accuracy/P/R/F1 + ماتریس درهم‌ریختگی |
| `/train/` | آموزش مجدد با داده اختیاری + بازگردانی نسخه قبل |

## بازتولید کامل

دیتاست‌های خام در ریپو نیستند (حجم/لایسنس). دانلود و در مسیرهای زیر بگذارید:

| منبع | مسیر |
|---|---|
| [persian-abusive-words](https://huggingface.co/datasets/AlirezaFzp/persian-abusive-words) (Apache-2.0) | `persian-abusive-words/{train,test}.csv` |
| [Naseza](https://github.com/amirivojdan/naseza) (CC0) | `candidates/naseza/naseza.json` |
| [ParsOffensive](https://github.com/golnaz76gh/pars-offensive-dataset) | `candidates/pars-offensive/ParsOffensive.xlsx` |
| [PHate](https://github.com/Zahra-D/Phate) | `candidates/phate/{train,val,test}_simple.csv` |
| [PHICAD](https://github.com/davardoust/PHICAD) | `candidates/phicad/PHICAD-part*.csv` |

```bash
python moderation/clean_abusive_words.py   # پاکسازی پایه
python moderation/build_merged.py          # ادغام ۵ منبع ایرانی (۱۰۹,۷۴۷)
python moderation/select_valuable.py       # گزینش ارزش‌محور → ۱۹,۸۰۰ نمونه
python moderation/make_eval_set.py         # مجموعه ارزیابی مستقل (۸۴ کامنت سایت)
python moderation/train_model_v4.py        # آموزش تک‌مدل + آستانه از validation
python moderation/report_utils.py          # گزارش کامل → report.json
```

## محدودیت‌های شناخته‌شده (صادقانه)

- مدل خطی/Bag-of-words است (الزام تکلیف: لجستیک) — سقف عملکرد پایین‌تر از ترنسفورمرهای فارسی (ParsBERT و مشابه)
- برچسب ۵ منبع با تعریف‌های متفاوت «نامناسب» ادغام شده — نوفه برچسب محتمل
- قاعده بلاک روی validation دقت ۹۸٫۹٪ داشت؛ روی ParsOffensive تست به ۸۸٪ افتاد (شکاف تعمیم واقعی و گزارش‌شده)
- فینگلیش و homoglyph پوشش داده نشده — کار آینده

## ⚠️ لایسنس داده‌ها

- هسته آزاد تجاری: **Apache-2.0** + **CC0** (Naseza)
- ParsOffensive / PHate / PHICAD لایسنس صریح ندارند — برای استفاده تجاری مجوز بگیرید یا با داده خودتان بازآموزی کنید
- کد این ریپو: MIT
