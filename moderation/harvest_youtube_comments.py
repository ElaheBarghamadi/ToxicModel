# -*- coding: utf-8 -*-
"""برداشت کامنت‌های واقعی فارسی از یوتیوب از طریق Invidious (بدون لاگین).

خروجی: data-website/collected_youtube_comments.csv (text, video_title, video_id, likes)
موضوعات: فوتبال/گرانی/سریال/موسیقی/نقد — بستری برای یافتن طعنه و توهین ملایم واقعی.
"""
import csv, html, json, re, subprocess, time, urllib.parse, urllib.request
from pathlib import Path

INST = 'https://invidious.f5.si'
# سرورهای سالم برای کامنت (به‌ترتیب تلاش) — catgirl: فرمت Invidious / ducks: فرمت Piped
COMMENT_SOURCES = [
    ('https://iv.catgirl.cloud', 'invidious'),
    ('https://pipedapi.ducks.party', 'piped'),
    ('https://invidious.f5.si', 'invidious'),
]
OUT = Path(__file__).resolve().parent.parent / 'data-website' / 'collected_youtube_comments.csv'
QUERIES = [
    'باخت تیم ملی ایران', 'استقلال پرسپولیس دربی', 'قیمت دلار امروز',
    'تحلیل بازی پرسپولیس', 'جدیدترین قسمت سریال ایرانی', 'داوری فوتبال ایران',
    'بازرسی کیفیت خودرو ایرانی', 'سفر به ترکیه گرانی',
]
PER_VIDEO_PAGES = 3   # هر صفحه ~۲۰ کامنت
SLEEP = 1.5

FA_RE = re.compile(r'[\u0600-\u06FF]')
LAT_RE = re.compile(r'[A-Za-z]')
URL_RE = re.compile(r'https?://|www\.|\.com|\.ir')


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode('utf-8'))


def clean(t):
    t = html.unescape(t or '')
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def persian_ok(t):
    if len(t) < 12 or len(t) > 400:
        return False
    if URL_RE.search(t):
        return False
    fa = len(FA_RE.findall(t))
    lat = len(LAT_RE.findall(t))
    return fa >= 6 and lat <= fa * 0.3


def ytdlp_search(q, n=3):
    """جستجوی ویدیو با yt-dlp (فقط شناسه/عنوان — دانلود نداریم)."""
    try:
        out = subprocess.run(
            ['yt-dlp', '--flat-playlist', '-J', f'ytsearch{n}:{q}'],
            capture_output=True, text=True, timeout=60)
        d = json.loads(out.stdout or '{}')
        return [(e['id'], clean(e.get('title'))) for e in d.get('entries', []) if e.get('id')]
    except Exception:
        return []


def fetch_comments(vid, pages=3):
    """کامنت‌های یک ویدیو؛ تلاش روی چند سرور با دو فرمت."""
    for base, kind in COMMENT_SOURCES:
        out, cont = [], None
        try:
            for _ in range(pages):
                if kind == 'invidious':
                    url = f'{base}/api/v1/comments/{vid}'
                    if cont:
                        url += '?continuation=' + urllib.parse.quote(cont)
                    d = get(url)
                    out += [{'text': clean(c.get('content')),
                             'likes': c.get('likeCount', 0)} for c in d.get('comments', [])]
                    cont = d.get('continuation')
                else:  # piped
                    url = f'{base}/comments/{vid}'
                    if cont:
                        url += '?nextpage=' + urllib.parse.quote(cont)
                    d = get(url)
                    out += [{'text': clean(c.get('commentText')),
                             'likes': c.get('likeCount', 0)} for c in d.get('comments', [])]
                    cont = d.get('nextpage')
                if not cont:
                    break
                time.sleep(SLEEP)
            if out:
                return out
        except Exception:
            continue
    return []


def main():
    videos = []
    seen_v = set()
    for q in QUERIES:
        found = [v for v in ytdlp_search(q) if v[0] not in seen_v and FA_RE.search(v[1] or '')]
        for v in found:
            seen_v.add(v[0])
            videos.append(v)
        print(f'🔎 «{q}» → {len(found)} ویدیو')
        time.sleep(SLEEP)

    rows, seen_t = [], set()
    for vid, title in videos:
        got = 0
        for c in fetch_comments(vid):
            t = c['text']
            if persian_ok(t) and t not in seen_t:
                seen_t.add(t)
                rows.append({'text': t, 'video_title': title[:60],
                             'video_id': vid, 'likes': c['likes']})
                got += 1
        print(f'  💬 {title[:40]} → {got} کامنت')
        time.sleep(SLEEP)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['text', 'video_title', 'video_id', 'likes'])
        w.writeheader()
        w.writerows(rows)
    print(f'\n✅ مجموع {len(rows)} کامنت یکتا → {OUT}')


if __name__ == '__main__':
    main()
