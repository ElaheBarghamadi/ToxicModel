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

## یافته جدید (شهریور ۱۴۰۵)
- **MirasIrony** — https://github.com/miras-tech/MirasText/tree/master/MirasIrony — ۲,۹۴۲ توییت ایرانی با برچسب طعنه/آیرونی (۱,۲۷۸ طعنه‌دار). زیپش **پسورد دارد**؛ پسورد فقط با ایمیل به behnamsabeti@gmail.com (نویسنده، LREC 2020). فایل زیپ در `candidates/miras-irony/`. نکته: برچسبش «طعنه» است نه «توهین» — فقط زیرمجموعه طعنه+توهین مفید است.

## برداشت کامنت واقعی از یوتیوب فارسی (شهریور ۱۴۰۵)
- اسکریپت: `moderation/harvest_youtube_comments.py` — جستجوی ویدیو با yt-dlp (ytsearch) + کامنت با سرورهای واسط.
- سرورهای سالم کامنت: `iv.catgirl.cloud/api/v1/comments/{id}` (Invidious) و `pipedapi.ducks.party/comments/{id}` (Piped). f5.si و بقیه Anubis/challenge دارند. ردیت/آپارات/خبری از سرور این محیط فیلترند (fetch_page برای ایرانی‌ها باز است ولی کامنت‌های تابناک/انتخاب AJAX است و در HTML نیست).
- خروجی فعلی: `data-website/collected_youtube_comments.csv` — ۵۲۱ کامنت یکتا از ۲۳ ویدیو (دربی/تیم ملی/دلار/سریال/سفر/خودرو).
