import os
from pathlib import Path

# ----- BASE DIR -----
BASE_DIR = Path(__file__).resolve().parent.parent


# ----- BASIC DJANGO SETTINGS -----
SECRET_KEY = 'd9g3%9z$w$+fp_eh7ud)b7osenwg*sz=+@1d4npv!b=@xay9ni'

DEBUG = True

ALLOWED_HOSTS = []


# ----- APPS -----
INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.staticfiles',
    'yourapp',   # your app
]


# ----- MIDDLEWARE -----
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
]


# ----- URLS / WSGI -----
ROOT_URLCONF = 'yourapp.urls'

WSGI_APPLICATION = 'yourapp.wsgi.application'


# ----- TEMPLATES -----
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # home.html, man.html, etc.
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.template.context_processors.static',
                'django.template.context_processors.media',
            ],
        },
    },
]


# ----- STATIC FILES -----
STATIC_URL = '/static/'


# ----- MEDIA (user uploaded photos + dataset images) -----
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'


# ----- DATABASE -----
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# ----- DEFAULT AUTO FIELD -----
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
