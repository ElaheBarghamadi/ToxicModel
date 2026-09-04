# -*- coding: utf-8 -*-
"""ویوهای سایت مدیریت محتوای فارسی."""
import csv
import json
import time
from io import BytesIO

from django.conf import settings
from django.http import FileResponse, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from . import batch_utils, model_utils, train_utils

DECISION_META = {
    'ok': ('✅ مجاز', 'انتشار فوری', 'ok'),
    'review': ('🔍 بازبینی', 'صف بازبینی انسانی', 'review'),
    'block': ('🚫 بلاک', 'مسدودسازی خودکار', 'block'),
}


def fmt(x, nd=3):
    return '—' if x is None else f'{x:.{nd}f}'.rstrip('0').rstrip('.')


# ---------------- داشبورد ----------------
def home(request):
    m = model_utils.load_metrics()
    v3 = m.get('v3', {})
    cfg = model_utils.model_config()
    per_test = v3.get('per_test', {})
    # جدول آستانه نسخه ۱ (تاریخی — روی تست تیون شده بود؛ در v3 فقط validation استفاده شد)
    thr_rows = []
    for t, row in sorted((m.get('threshold_table') or {}).items(), key=lambda kv: float(kv[0])):
        thr_rows.append({'thr': fmt(float(t), 2), **{k: fmt(v) for k, v in row.items()}})
    stats = model_utils.train_data_stats()
    src_fa = {'tweets': 'توییتر', 'naseza': 'تلگرام',
              'pars_offensive': 'اینستاگرام', 'phate': 'PHate',
              'phicad': 'اینستاگرام (PHICAD)'}
    ctx = {
        'rows': [
            {'name': n, 'src_fa': src_fa.get(n, n), **d} for n, d in per_test.items()],
        'thr_rows': thr_rows,
        'cfg': cfg,
        'train_total': v3.get('train_size_v2', stats['total']),
        'stats': stats,
    }
    for row in ctx['rows']:
        if 'F1' in row:
            row['f1_pct'] = round(row['F1'] * 100, 1)
            row['ci_lo_pct'] = round(row['F1_ci95'][0] * 100, 1)
            row['ci_span_pct'] = round((row['F1_ci95'][1] - row['F1_ci95'][0]) * 100, 1)
    return render(request, 'core/home.html', ctx)


# ---------------- گزارش کامل ----------------
def report_page(request):
    try:
        rep = json.load(open(settings.MODERATION_DIR / 'report.json', encoding='utf-8'))
    except Exception:
        rep = None
    return render(request, 'core/report.html', {'r': rep})


# ---------------- تست تکی ----------------
@csrf_exempt
def single_test(request):
    results = None
    text = ''
    cfg = model_utils.model_config()
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if text:
            lines = [l.strip() for l in text.splitlines() if l.strip()][:50]
            results = []
            for r in model_utils.predict_rows(lines):
                icon, desc, cls = DECISION_META[r['decision']]
                results.append({**r, 'icon': icon, 'desc': desc, 'css': cls})
    return render(request, 'core/test.html',
                  {'results': results, 'results_json': results or [], 'text': text, 'cfg': cfg})


# ---------------- تست با فایل ----------------
@csrf_exempt
def batch_test(request):
    ctx = {'cfg': model_utils.model_config()}
    if request.method == 'POST' and request.FILES.get('file'):
        f = request.FILES['file']
        try:
            rows, text_col, label_col = batch_utils.parse_file(f)
        except Exception as e:
            ctx['error'] = f'خطا در خواندن فایل: {e}'
            rows = []
        if rows:
            preds = model_utils.predict_rows([t for t, _ in rows])
            table = []
            for (t, lab), p in zip(rows, preds):
                icon, desc, cls = DECISION_META[p['decision']]
                table.append({'text': t, 'label': lab, 'decision': p['decision'],
                              'icon': icon, 'css': cls,
                              'prob': p['prob_abusive'],
                              'match': None if lab is None else (
                                  int(p['decision'] in ('review', 'block')) == lab)})
            labeled = [r for r in table if r['label'] is not None]
            ctx.update({
                # کل داده به‌صورت JSON — رندر تکه‌ای در مرورگر (لیزی‌لود)
                'rows_json': table[:500],
                'n_total': len(table),
                'truncated': len(table) > 500,
                'has_label': bool(labeled),
                'metrics': batch_utils.evaluate(
                    [(r['text'], r['label'], r['decision']) for r in labeled]) if labeled else None,
                'text_col': text_col, 'label_col': label_col,
                'decision_counts': {d: sum(1 for r in table if r['decision'] == d)
                                    for d in ('ok', 'review', 'block')},
            })
            # ذخیره خروجی کامل برای دانلود
            with open(settings.MEDIA_DIR / 'batch_results.csv', 'w',
                      encoding='utf-8-sig', newline='') as fh:
                out = csv.writer(fh)
                out.writerow(['text', 'prob_abusive', 'decision', 'label', 'correct'])
                for r in table:
                    out.writerow([r['text'], r['prob'], r['decision'],
                                  '' if r['label'] is None else r['label'],
                                  '' if r['match'] is None else int(r['match'])])
            ctx['download_ready'] = True
    return render(request, 'core/batch.html', ctx)


def batch_download(request, fname):
    if fname != 'batch_results.csv':
        return HttpResponse('نامعتبر', status=404)
    return FileResponse(open(settings.MEDIA_DIR / fname, 'rb'), as_attachment=True,
                        filename='batch_results.csv')


def batch_sample(request):
    """نمونه CSV لیبل‌دار برای امتحان سریع."""
    sample = [
        ('عالی بود ممنون از نویسنده', 0),
        ('قیمت مناسبه ولی کیفیت دوام نداره', 0),
        ('پشتیبانی جواب نمی‌ده، ناراضی‌ام', 0),
        ('نویسنده‌ش خودش یه الاغ بی‌سواده', 1),
        ('همه کارمندای این سایت دزد هستن', 1),
        ('بریم کیر تو این سایت خراب شد', 1),
        ('بی غیرت هر کی این خبر رو نوشته', 1),
        ('ممنون از توضیحات کاملتون', 0),
    ]
    import io as _io
    sio = _io.StringIO()
    w = csv.writer(sio)
    w.writerow(['text', 'label'])
    w.writerows(sample)
    buf = BytesIO(sio.getvalue().encode('utf-8-sig'))
    return FileResponse(buf, as_attachment=True, filename='sample.csv')


# ---------------- آموزش ----------------
def train_page(request):
    st = train_utils.read_status()
    extra_files = [p.name for p in settings.EXTRA_DATA_DIR.glob('*.csv')]
    last = st.get('summary') or {}
    return render(request, 'core/train.html', {
        'status': st, 'log': train_utils.log_tail(),
        'extra_files': extra_files, 'last_summary': last,
        'stats': model_utils.train_data_stats(),
        'config': model_utils.model_config(),
        'archives': [p.name for p in model_utils.archived_models()[:3]]})


@csrf_exempt
def train_upload_data(request):
    if request.method == 'POST' and request.FILES.get('file'):
        f = request.FILES['file']
        try:
            rows, tc, lc = batch_utils.parse_file(f)
            labeled = [(t, l) for t, l in rows if l is not None]
            if not labeled:
                return render(request, 'core/train.html', {
                    'status': train_utils.read_status(),
                    'log': train_utils.log_tail(),
                    'extra_files': [p.name for p in settings.EXTRA_DATA_DIR.glob('*.csv')],
                    'last_summary': {}, 'stats': model_utils.train_data_stats(),
                    'upload_error': 'ستون label با مقادیر ۰/۱ پیدا نشد.'})
            name = f'extra_{int(time.time())}.csv'
            with open(settings.EXTRA_DATA_DIR / name, 'w', encoding='utf-8') as out:
                w = csv.writer(out)
                w.writerow(['text', 'label'])
                w.writerows(labeled)
            return render(request, 'core/train.html', {
                'status': train_utils.read_status(),
                'log': train_utils.log_tail(),
                'extra_files': [p.name for p in settings.EXTRA_DATA_DIR.glob('*.csv')],
                'last_summary': {},
                'stats': model_utils.train_data_stats(),
                'upload_ok': f'{len(labeled)} نمونه لیبل‌دار ذخیره شد ({name}) — حالا آموزش را شروع کنید.'})
        except Exception as e:
            pass
    return train_page(request)


@csrf_exempt
def train_start(request):
    if request.method == 'POST':
        started = train_utils.start_training()
        st = train_utils.read_status()
        return JsonResponse({'started': started, 'status': st})
    return JsonResponse({'error': 'POST فقط'}, status=405)


@csrf_exempt
def train_rollback(request):
    """بازگردانی آخرین نسخه بایگانی‌شده مدل."""
    if request.method == 'POST':
        ok, msg = model_utils.rollback_model()
        return JsonResponse({'ok': ok, 'message': msg})
    return JsonResponse({'error': 'POST فقط'}, status=405)


def status_json(request):
    st = train_utils.read_status()
    if not st.get('running') and st.get('stage') == 'done':
        model_utils.reload_models()   # مدل تازه‌آموزیده را بارگذاری کن
    return JsonResponse({'status': st, 'log': train_utils.log_tail(25)})
