# 🛡️ ToxicModel — تشخیص کامنت نامناسب فارسی

سیستم تشخیص محتوای نامناسب (فحش و توهین) در کامنت‌های فارسی + رابط وب جنگو.

**English TL;DR:** Persian toxic/offensive comment detection — two-model architecture
(narrow high-precision profanity model + broad 5-source model) with a Django web app
for single/batch testing (auto-scoring when labels present) and in-app retraining.

---

## نتایج

| مجموعه آزمون | دامنه | v1 — F1 | v2 — F1 |
|---|---|---|---|
| توییتر (Persian Abusive Words) | توییت | **0.98** | 0.83* |
| Naseza | تلگرام | 0.66 | **0.88** |
| ParsOffensive | کامنت اینستاگرام | 0.53 | **0.79** |
| PHate | توییتر | 0.63 | **0.76** |
| PHICAD | کامنت اینستاگرام | 0.61 | **0.92** |

\* تعریف «نامناسب» در v2 گسترده‌تر است (توهین ملایم هم شامل می‌شود)؛ برای همین معماری دو-مدلی استفاده می‌شود.

### معماری تصمیم

```
p1 = مدل فحش سخت‌گیر (model_v1.joblib)     p2 = مدل پهن ۵-منبعی (model.joblib)

p1 ≥ 0.9  و  p2 ≥ 0.7   →  🚫 block   (مسدودسازی خودکار)
p2 ≥ 0.5                →  🔍 review  (صف بازبینی انسانی)
در غیر این صورت          →  ✅ ok      (انتشار فوری)
```

- ویژگی‌ها: char TF-IDF (2-5گرم) روی «متن + نسخهٔ بدون فاصله» (ضف فحش حرف‌جدا like «ک ی ر») ∪ word TF-IDF
- سرعت: ~۲ میلی‌ثانیه بر کامنت، بدون GPU — حجم هر مدل ۱۳ تا ۴۰ مگابایت

## ساختار ریپو

```
moderation/            # مدل‌ها و پایپ‌لاین
├── model.joblib           ← v2 (مدل پهن، ۱۰۹,۷۴۷ نمونه از ۵ منبع)
├── model_v1.joblib        ← v1 (مدل سخت‌گیر فحش)
├── moderate.py            ← API استنتاج: predict() → ok/review/block
├── mod_text.py            ← نرمال‌سازی فارسی (یونیکد، اعراب، کشیدگی…)
├── train_model.py         ← آموزش v1
├── train_model_v2.py      ← آموزش v2
├── clean_abusive_words.py ← پاکسازی دیتاست پایه (dedupe + رفع نشت)
├── build_merged.py        ← ساخت دیتاست ادغام‌شده ۵ منبع
├── train_web.py           ← آموزش از طریق وب (با گزارش پیشرفت)
└── metrics.json           ← ارزیابی‌ها

modsite/               # رابط وب جنگو (RTL فارسی)
├── core/                # ویوها، پردازش فایل، اجرای آموزش
├── templates/core/      # داشبورد / تست تکی / تست فایل / آموزش
└── requirements.txt
```

## اجرای سایت

```bash
pip install -r modsite/requirements.txt
cd modsite && python manage.py runserver
# http://127.0.0.1:8000
```

صفحات: `/` داشبورد متریک‌ها + توضیح · `/test/` تست تک‌کامنت · `/batch/` تست با فایل
(CSV/XLSX؛ اگر ستون `label` داشت نمرات Accuracy/P/R/F1 و ماتریس درهم‌ریختگی را می‌دهد) ·
`/train/` آموزش مجدد با داده اختیاری شما

## بازتولید کامل از صفر

دیتاست‌های خام در ریپو نیستند (حجم/لایسنس) — این‌ها را دانلود و در مسیرهای زیر بگذارید:

| منبع | کجا بگذارید |
|---|---|
| [persian-abusive-words](https://huggingface.co/datasets/AlirezaFzp/persian-abusive-words) (Apache-2.0) | `persian-abusive-words/{train,test}.csv` |
| [Naseza](https://github.com/amirivojdan/naseza) (CC0) | `candidates/naseza/naseza.json` |
| [ParsOffensive](https://github.com/golnaz76gh/pars-offensive-dataset) | `candidates/pars-offensive/ParsOffensive.xlsx` |
| [PHate](https://github.com/Zahra-D/Phate) | `candidates/phate/{train,val,test}_simple.csv` |
| [PHICAD](https://github.com/davardoust/PHICAD) | `candidates/phicad/PHICAD-part*.csv` |

سپس:

```bash
python moderation/clean_abusive_words.py   # پاکسازی دیتاست پایه
python moderation/build_merged.py          # ساخت data-merged/
python moderation/train_model.py           # آموزش v1
python moderation/train_model_v2.py        # آموزش v2
```

## ⚠️ لایسنس داده‌ها

- هسته آزاد تجاری: **Apache-2.0** (دیتاست پایه) + **CC0** (Naseza)
- ParsOffensive / PHate / PHICAD لایسنس صریح ندارند (مصنوع تحقیقاتی) — برای استفاده تجاری از نویسندگان مجوز بگیرید یا با داده خودتان بازآموزی کنید (`/train/` سایت همین کار را می‌کند)
- کد این ریپو: MIT
