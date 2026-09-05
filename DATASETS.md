# منابع داده (فقط لینک‌ها — دیتاست‌ها در ریپو نیستند)

| # | دیتاست | لینک | دامنه | لایسنس |
|---|---|---|---|---|
| ۱ | Persian Abusive Words | https://huggingface.co/datasets/AlirezaFzp/persian-abusive-words | توییتر ایرانی | Apache-2.0 |
| ۲ | Naseza (ناسزا) | https://github.com/amirivojdan/naseza | تلگرام ایرانی (سیاسی/ورزشی) | CC0 |
| ۳ | ParsOffensive | https://github.com/golnaz76gh/pars-offensive-dataset | کامنت اینستاگرام ایرانی | اعلام‌نشده |
| ۴ | PHate | https://github.com/Zahra-D/Phate | توییتر ایرانی (AAAI-24) | اعلام‌نشده |
| ۵ | PHICAD | https://github.com/davardoust/PHICAD | کامنت اینستاگرام ایرانی | اعلام‌نشده |

## رد‌شده‌ها
- ~~Farsi-Dari (Kaggle/yasinhazara)~~ — دری/افغانی، خارج دامنه ایرانی
- ~~Detecting-toxic-comments (HF)~~ — انگلیسی (Jigsaw)
- ~~Pars-OFF~~ — فایل rar، ناموفق در استخراج

## مسیرگذاری بعد از دانلود (برای build_merged.py)
```
persian-abusive-words/{train,test}.csv
candidates/naseza/naseza.json
candidates/pars-offensive/ParsOffensive.xlsx
candidates/phate/{train,val,test}_simple.csv
candidates/phicad/PHICAD-part{1,2}.csv
```
