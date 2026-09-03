# ناظر — سامانه مدیریت محتوای فارسی (Django)

رابط وب برای مدل تشخیص کامنت نامناسب فارسی (نسخه ۲، دو-مدلی).

## اجرا

```bash
cd modsite
pip install -r requirements.txt
python manage.py runserver 0.0.0.0:8000
```

## صفحات

| مسیر | کار |
|---|---|
| `/` | داشبورد: متریک‌های ۵ مجموعه آزمون + توضیح ساده هر متریک + جدول اثر آستانه + معماری |
| `/test/` | تست تک‌کامنت (چند خط = چند کامنت) → احتمال + تصمیم ok/review/block |
| `/batch/` | آپلود CSV/XLSX با ستون `text` (و `label` اختیاری) → پیش‌بینی + اگر لیبل داشت: Accuracy/P/R/F1/ماتریس درهم‌ریختگی + دانلود نتایج. نمونه آماده: `/batch/sample/` |
| `/train/` | آموزش مجدد: آپلود داده اضافی اختیاری (text,label) + دکمه شروع با نوار پیشرفت زنده (~۱ دقیقه) |

## معماری تصمیم

```
p1 = مدل فحش سخت‌گیر (model_v1.joblib)   p2 = مدل پهن ۵-منبعی (model.joblib)

p1≥0.9 و p2≥0.7 → block   |   p2≥0.5 → review   |   در غیر این صورت → ok
```

## ساختار

```
modsite/
├── manage.py
├── requirements.txt
├── modsite/            # تنظیمات و urls
├── core/               # ویوها + پردازش فایل + اجرای آموزش
└── templates/core/     # قالب‌های RTL فارسی
```

وابستگی به مدل‌ها: پوشه `/home/user/moderation` (model.joblib ، model_v1.joblib ، mod_text.py) باید در کنار پروژه باشد — مسیرش در `settings.py` تنظیم شده.

آموزش مجدد وب: `moderation/train_web.py` (وضعیت زنده در `moderation/train_status.json`).
