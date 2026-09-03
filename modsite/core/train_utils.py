# -*- coding: utf-8 -*-
"""اجرای آموزش در پس‌زمینه + خواندن وضعیت."""
import json
import subprocess
import time
from pathlib import Path

from django.conf import settings

MOD = Path(settings.MODERATION_DIR)
SCRIPT = MOD / 'train_web.py'
STATUS = Path(settings.TRAIN_STATUS_FILE)
LOG = Path(settings.TRAIN_LOG_FILE)


def read_status():
    try:
        return json.load(open(STATUS, encoding='utf-8'))
    except Exception:
        return {'running': False, 'stage': 'idle', 'pct': 0, 'message': 'آماده'}


def is_running():
    st = read_status()
    if not st.get('running'):
        return False
    # اگر بیش از ۳۰ دقیقه است بدون به‌روزرسانی، مرده تلقی می‌شود
    if time.time() - st.get('heartbeat', 0) > 1800:
        return False
    return True


def log_tail(n=30):
    try:
        lines = open(LOG, encoding='utf-8', errors='replace').read().strip().splitlines()
        return '\n'.join(lines[-n:])
    except Exception:
        return ''


def start_training():
    """اسکریپت آموزش را در پس‌زمینه اجرا می‌کند."""
    if is_running():
        return False
    logf = open(LOG, 'w', encoding='utf-8')
    subprocess.Popen(
        ['python', str(SCRIPT)],
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=str(MOD), start_new_session=True,
    )
    return True
