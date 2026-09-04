# 🛡️ ToxicModel — تشخیص کامنت نامناسب فارسی (v3)

سیستم تشخیص محتوای نامناسب (فحش و توهین) در کامنت‌های فارسی + رابط وب جنگو.

**English TL;DR:** Persian toxic-comment detection — dual **Logistic Regression**
architecture (narrow high-precision profanity model + broad 5-source model),
validation-derived thresholds, bootstrap CIs, versioned/rollback-safe retraining,
plus a Django app for single/batch testing (auto-scoring when labels exist).

> **الزام تکلیف:** الگوریتم مدل = **رگرسیون لجستیک (Logistic Regression)** — هر دو مدل.

---

## نتایج نسخه ۳ (تست‌های دست‌نخورده، آستانه‌ها فقط از validation)

**داده آموزش: ۱۹٬۸۰۰ نمونه گزینش‌شده** (محدودیت زیر ۲۰ هزار — گزینش ارزش‌محور با `select_valuable.py`)

| مجموعه آزمون | دامنه | F1 | فاصله اطمینان ۹۵٪ | بلاک (دقت/بازیابی) |
|---|---|---|---|---|
| توییتر | Persian Abusive Words | 0.733 | [0.702, 0.762] | 1.000 / 0.239 |
| Naseza | تلگرام | 0.821 | [0.791, 0.850] | 1.000 / 0.266 |
| ParsOffensive | کامنت اینستاگرام | 0.739 | [0.710, 0.769] | 0.983 / 0.123 |
| PHate | توییتر | 0.694 | [0.660, 0.727] | 1.000 / 0.069 |
| PHICAD | کامنت اینستاگرام | 0.879 | [0.873, 0.885] | 0.998 / 0.092 |
| **همه با هم (۱۹٬۵۱۷)** | — | **0.849** | [0.843, 0.854] | — |

> مقایسه با آموزش روی کل ۱۰۹٬۷۴۷ نمونه: F1 کلی ۰٫۸۹۸ → ۰٫۸۴۹ — هزینه محدودیتِ داده فقط ~۰٫۰۵ F1 است.
> ترکیب گزینش (انتخاب‌شده با آزمایش روی validation ثابت): ۱٬۰۰۰ سخت + ۱٬۰۰۰ مرزی + ۱۷٬۸۰۰ طبقه‌بندی‌شده متناسب با سهمیه توییتر ۵٬۰۰۰. توجه: آموزش عمدتاً روی نمونه‌های «سخت»ِ برچسب‌نویزی، کالیبراسیون را خراب می‌کند (آزمایش شد و رد شد).

### معماری تصمیم

```
p1 = مدل فحش سخت‌گیر (model_v1.joblib، LR C=10)   p2 = مدل پهن ۵-منبعی (model.joblib، LR C=4)

p1 ≥ 0.95  و  p2 ≥ 0.95  →  🚫 block   (انتخاب‌شده روی validation؛ دقت ۹۸-۱۰۰٪ روی تست‌ها)
p2 ≥ 0.517               →  🔍 review
در غیر این صورت           →  ✅ ok
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
├── model.joblib / model_v1.joblib   ← مدل‌های فعال (هر دو LogisticRegression)
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
python moderation/build_merged.py          # ادغام ۵ منبع (۱۰۹,۷۴۷)
python moderation/select_valuable.py       # گزینش ارزش‌محور → ۱۹,۸۰۰ نمونه
python moderation/train_model_v3.py        # آموزش نهایی (LR×2 + validation + CI)
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
