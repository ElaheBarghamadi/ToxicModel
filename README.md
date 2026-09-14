# ToxicModel — مدل تشخیص محتوای نامناسب فارسی

مدل یادگیری ماشین برای تشخیص فحش و توهین در کامنت‌های فارسی (ایرانی).
تک‌مدل **LogisticRegression** روی ویژگی‌های TF-IDF (کاراکتری + واژه‌ای، با نرمال‌سازی فارسی) — تصمیم دوتایی: ✅ ok / 🚫 block.

> این مخزن فقط مدل است: داده‌ها و اسکریپت‌های آموزش. (رابط وب حذف شد.)

## مدل فعال (v4-ir)

| مشخصه | مقدار |
|---|---|
| الگوریتم | LogisticRegression (C=10, class_weight=balanced) |
| داده آموزش | ۱۶٬۰۰۰ نمونه — **همه OOF-تاییدشده** (۴۸۶ برچسب‌مشکوک + ۱۱٬۲۳۴ تناقض‌دار حذف شده) |
| آستانه بلاک | p ≥ 0.467 (از validation، روی منحنی PR) |
| نرمال‌سازی | یکسان‌سازی ی/ک، ارقام، فینگلیش→فارسی، له‌کردن سجاوندی جداکننده |

### عملکرد (F1)

| تست | توییتر | تلگرام (ناسزا) | ParsOffensive | PHate | PHICAD | کامنت سایت (۸۴) | **کل (mixed)** |
|---|---|---|---|---|---|---|---|
| F1 | 0.706 | 0.822 | 0.726 | 0.699 | 0.896 | 0.727 | **0.858 [±0.006]** |

### مقایسه الگوریتم‌ها (پروتکل یکسان)

| مدل | mixed F1 |
|---|---|
| **LogisticRegression (استقرار)** | 0.860 |
| LinearSVC | 0.860 |
| MultinomialNB | 0.853 |
| آنسامبل رأی‌گیری نرم | 0.864 (تفاوت در مرز CI — معنادار نیست) |

### ارزیابی واقع‌گرایانه
- **۵۲۱ کامنت واقعی یوتیوب فارسی** (دست‌لیبل): F1 = 0.336 — سخت‌ترین بنچمارک؛ خطاها: نقد سیاسیِ سالم اشتباه بلاک می‌شود (سوگیری داده‌های عمومی) و توهین مؤدبانه/واژگان جدید رد می‌شود. جزئیات: `metrics.json`
- ۹۴۹ کامنت واقعی طلایی (`data-website/real_gold_all.csv`) برای دورهای بعدی آموزش.

## خط تولید (بازتولید از صفر)

```bash
# ۱) دانلود منابع از لینک‌های DATASETS.md → candidates/
python3 moderation/clean_abusive_words.py     # پاک‌سازی واژه‌نامه توهین
python3 moderation/build_merged.py            # ادغام → data-merged/train_merged.csv (۱۰۹k)
python3 moderation/select_valuable_v3.py      # گزینش کیفیت‌محور → ۱۶k تاییدشده
python3 moderation/train_model_v4.py          # آموزش + آستانه → model.joblib
python3 moderation/report_utils.py            # گزارش → report.json
python3 moderation/run_external_eval.py       # ارزیابی مستقل (مسیر استنتاج واقعی)
```

## استنتاج

```python
import sys; sys.path.insert(0, 'moderation')
from moderate import predict
predict(['برو دنجی'])   # → [{'decision': 'block', 'p': ...}, ...]
```

## ساختار

```
moderation/    مدل + آموزش + گزینش + ارزیابی + نرمال‌سازی
data-merged/   استخر ۱۰۹k + آموزش ۱۶k + ۵ تست خارجی (gitignored — لایسنس منابع)
data-website/  real_gold_all.csv (۹۴۹ طلایی واقعی)
eval/          وب‌سایت ۸۴ + یوتیوب واقعی ۵۲۱
tamrin2/       تمرین ۲ کامل: مدل تعدیل کامنت pycourse.ir — داده ۱۷۴۹ / آموزش / گزارش + پوشه‌ی تحویل Elahe-Barghamadi/
DATASETS.md    لینک منابع + سابقه
```
