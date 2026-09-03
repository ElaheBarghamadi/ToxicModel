"""تنظیمات سایت مدیریت محتوای فارسی."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # modsite/
ROOT_DIR = BASE_DIR.parent                           # /home/user

SECRET_KEY = 'modsite-local-secret-key-change-me'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'core',
]

MIDDLEWARE = [
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'modsite.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': []},
}]

WSGI_APPLICATION = 'modsite.wsgi.application'

DATABASES = {}   # پایگاه‌داده لازم نیست — مدل‌ها joblib هستند

LANGUAGE_CODE = 'fa'
TIME_ZONE = 'Asia/Baku'
USE_I18N = False

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# مسیرهای پروژه مدل
MODERATION_DIR = ROOT_DIR / 'moderation'
DATA_MERGED_DIR = ROOT_DIR / 'data-merged'
MEDIA_DIR = BASE_DIR / 'media'
MEDIA_DIR.mkdir(exist_ok=True)
EXTRA_DATA_DIR = MODERATION_DIR / 'extra_data'
EXTRA_DATA_DIR.mkdir(exist_ok=True)
TRAIN_STATUS_FILE = MODERATION_DIR / 'train_status.json'
TRAIN_LOG_FILE = MODERATION_DIR / 'train_web.log'
