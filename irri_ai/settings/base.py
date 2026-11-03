"""
Base settings for irri_ai project.
Environment-specific overrides live in dev.py / prod.py.
"""
from pathlib import Path
from decouple import Csv, config

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = config("SECRET_KEY", default="dev-insecure-secret-key-change-me")
DEBUG = config(
    "DEBUG",
    default=True,
    cast=lambda value: str(value).strip().lower()
    not in {"0", "false", "no", "off", "release", "prod", "production"},
)
ALLOWED_HOSTS = config("ALLOWED_HOSTS", default="127.0.0.1,localhost", cast=Csv())

# Django 4.0+ checks the request Origin against this list (scheme + host) for
# POST forms. Any HTTPS domain the app is served from must be listed here or
# every form submit fails with "CSRF verification failed / Origin checking".
CSRF_TRUSTED_ORIGINS = config(
    "CSRF_TRUSTED_ORIGINS",
    default="https://paddymitra-dev.indevconsultancy.in",
    cast=Csv(),
)

# --- Applications ----------------------------------------------------------

DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
]

THIRD_PARTY_APPS = [
    "widget_tweaks",
]

LOCAL_APPS = [
    "core",
    "accounts",
    "documents",
    "keywords",
    "catalogues",
    "hooks",
    "queries",
    "dashboard",
    "image_bank",
    "api_endpoints",
]

# --- Image bank (CLIP visual knowledge) -----------------------------------
IMAGE_BANK_DISTANCE_MAX = 0.35  # cosine distance; lower is better. Above this -> "low confidence"
IMAGE_BANK_TOP_K = 5

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "irri_ai.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "core.context_processors.nav",
            ],
        },
    },
]

WSGI_APPLICATION = "irri_ai.wsgi.application"
ASGI_APPLICATION = "irri_ai.asgi.application"

# --- Database (MySQL via XAMPP) --------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": config("DB_NAME", default="irri_ai"),
        "USER": config("DB_USER", default="root"),
        "PASSWORD": config("DB_PASSWORD", default=""),
        "HOST": config("DB_HOST", default="127.0.0.1"),
        "PORT": config("DB_PORT", default="3306"),
        "OPTIONS": {
            "charset": "utf8mb4",
            "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
        },
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Auth ------------------------------------------------------------------

AUTH_USER_MODEL = "accounts.User"
LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "dashboard:home"
LOGOUT_REDIRECT_URL = "accounts:login"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# --- i18n ------------------------------------------------------------------

LANGUAGE_CODE = "en-in"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
# Set False because XAMPP MySQL ships without the timezone tables, so
# TruncMonth + CONVERT_TZ fails. Storing naive local timestamps is fine for a single-region app.
USE_TZ = False

# --- Static / media --------------------------------------------------------

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DATA_UPLOAD_MAX_MEMORY_SIZE = 25 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 25 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FILES = 5000

# --- Chroma / embeddings / LLM --------------------------------------------

CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Docling: structure-aware PDF chunks produced OFFLINE by docling_ingest.py and
# cached as one JSON per PDF (keyed by the PDF's content hash). When a cache file
# exists for a PDF it is preferred over Sherpa; otherwise chunking falls back to
# Sherpa, then pypdf. Docling itself is NOT a runtime dependency of this app —
# only the cache is read. See docling_ingest.py and catalogues/services/chunking.py.
USE_DOCLING = config("USE_DOCLING", default=True, cast=bool)
DOCLING_CACHE_DIR = config("DOCLING_CACHE_DIR", default=str(BASE_DIR / "docling_cache"))

LLM_SHERPA_API_URL = config(
    "LLM_SHERPA_API_URL", default=""
)
RRF_K = config("RRF_K", default=60, cast=int)
USE_CROSS_ENCODER = config("USE_CROSS_ENCODER", default=False, cast=bool)
CROSS_ENCODER_MODEL = config(
    "CROSS_ENCODER_MODEL", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

OPENAI_API_KEY = config("OPENAI_API_KEY", default="")
OPENAI_EMBED_MODEL = config("OPENAI_EMBED_MODEL", default="text-embedding-3-small")
OPENAI_CHAT_MODEL = config("OPENAI_CHAT_MODEL", default="gpt-4o-mini")
OPENAI_STT_MODEL = config("OPENAI_STT_MODEL", default="whisper-1")
OPENAI_TTS_MODEL = config("OPENAI_TTS_MODEL", default="tts-1")
OPENAI_TTS_VOICE = config("OPENAI_TTS_VOICE", default="alloy")
NARAKEET_KEY = config("NARAKEET_KEY", default="")
GEMINI_API_KEY = config("GEMINI_API_KEY", default="")
GEMINI_TRANSLATE_MODEL = config("GEMINI_TRANSLATE_MODEL", default="gemini-3.5-flash-lite")

USE_LOCAL_EMBEDDINGS = config("USE_LOCAL_EMBEDDINGS", default=False, cast=bool)
GLOSSARY_AUTO_LEARN = config("GLOSSARY_AUTO_LEARN", default=True, cast=bool)

# Load the local Odia Wav2Vec2 model at startup instead of on first use. It is
# the FALLBACK for Sarvam STT and costs ~1.2 GB resident whether or not it ever
# runs, so it is off by default; see queries/apps.py.
PRELOAD_LOCAL_STT = config("PRELOAD_LOCAL_STT", default=False, cast=bool)
STT_TRANSCRIPT_CORRECTION = config("STT_TRANSCRIPT_CORRECTION", default=True, cast=bool)
ODIA_SEMANTIC_REPAIR = config("ODIA_SEMANTIC_REPAIR", default=True, cast=bool)
LOG_RAW_TRANSLATION = config("LOG_RAW_TRANSLATION", default=False, cast=bool)


# --- Odia STT --------------------------------------------------------------
# Primary path: Sarvam AI hosted ASR (fast, Indic-tuned). Set SARVAM_API_KEY
# to enable. If blank or the API fails, the request falls back to the local
# Wav2Vec2 model (`Harveenchadha/odia_large_wav2vec2`) — slower but offline.
SARVAM_API_KEY = config("SARVAM_API_KEY", default="")
SARVAM_STT_URL = config("SARVAM_STT_URL", default="https://api.sarvam.ai/speech-to-text")
SARVAM_STT_MODEL = config("SARVAM_STT_MODEL", default="saaras:v3")
SARVAM_STT_LANGUAGE_CODE = config("SARVAM_STT_LANGUAGE_CODE", default="od-IN")
SARVAM_STT_MODE = config("SARVAM_STT_MODE", default="transcribe")
FORCE_LOCAL_ODIA_STT = config("FORCE_LOCAL_ODIA_STT", default=False, cast=bool)

# --- Odia text-to-speech ---------------------------------------------------
# Sarvam AI speaks the answer, through the SAME SARVAM_API_KEY as the STT above.
# Pace is a little under natural speed on purpose: the answers carry doses and
# stage names and the farmer hears them once. See queries/services/narakeet_tts.py.
SARVAM_TTS_MODEL = config("SARVAM_TTS_MODEL", default="bulbul:v3")
SARVAM_TTS_SPEAKER = config("SARVAM_TTS_SPEAKER", default="shubh")
SARVAM_TTS_LANGUAGE_CODE = config("SARVAM_TTS_LANGUAGE_CODE", default="od-IN")
SARVAM_TTS_PACE = config("SARVAM_TTS_PACE", default=0.9, cast=float)

# --- WhatsApp / Picky Assist ----------------------------------------------

PICKY_ASSIST_TOKEN = config("PICKY_ASSIST_TOKEN", default="")
PICKY_ASSIST_WEBHOOK_SECRET = config("PICKY_ASSIST_WEBHOOK_SECRET", default="")
PICKY_ASSIST_SEND_URL = config(
    "PICKY_ASSIST_SEND_URL", default="https://api.pickyassist.com/v1/outbound/send"
)
# The account's application id, sent as "application" in every outbound body.
# There is no sensible default: the gateway accepts a send without it and then
# delivers nothing, so the sender refuses rather than fail silently.
PICKY_ASSIST_APPLICATION = config("PICKY_ASSIST_APPLICATION", default="")
PICKY_ASSIST_PRIORITY = config("PICKY_ASSIST_PRIORITY", default="1")

# The inbound webhook is a PUBLIC url — the farmer's gateway is the only thing
# that should be able to POST to it. Without the shared secret, anyone who
# guesses the path can spend our OpenAI budget and write rows into Query. So a
# request with no matching secret is refused: set PICKY_ASSIST_WEBHOOK_SECRET
# and configure the same value as a custom header (X-Webhook-Secret) or query
# parameter (?secret=) on the Picky Assist side.
# The escape hatch is explicit and off by default, because "it stopped working
# after we set a secret" is a much better failure than a webhook nobody knew
# was open.
WHATSAPP_ALLOW_UNAUTHENTICATED_WEBHOOK = config(
    "WHATSAPP_ALLOW_UNAUTHENTICATED_WEBHOOK", default=False, cast=bool)

# The gateway retries when it does not get a fast 200, and answering a farmer
# twice is worse than answering slowly. The webhook therefore acks immediately
# and runs the pipeline on these threads. Threads, not Celery: there is no
# broker in this deployment, and one worker per concurrent farmer is enough at
# this volume. A message whose thread dies with the process is picked up again
# by `manage.py whatsapp_retry`.
WHATSAPP_WORKERS = config("WHATSAPP_WORKERS", default=4, cast=int)

# How long a repeated gateway message id is treated as the SAME message.
WHATSAPP_DEDUPE_WINDOW_HOURS = config(
    "WHATSAPP_DEDUPE_WINDOW_HOURS", default=24, cast=int)

# WhatsApp rejects a body over 4096 characters, so a long answer is sent as
# several messages rather than truncated at the gateway.
WHATSAPP_MAX_CHARS = config("WHATSAPP_MAX_CHARS", default=3800, cast=int)

# Voice replies. The TTS file is already produced and stored for every hook
# message; the gateway needs a URL it can fetch, which only exists once this is
# set to the public origin serving MEDIA_URL (e.g. https://paddymitra.example.org).
# Left blank, the answer goes out as text only — never as a broken link.
PUBLIC_BASE_URL = config("PUBLIC_BASE_URL", default="").rstrip("/")
WHATSAPP_SEND_VOICE = config("WHATSAPP_SEND_VOICE", default=True, cast=bool)

# Keep the farmer's own voice note, not just its transcript. It is stored on
# the Query (media/audio/in/) and is the only way to check a transcript that
# reads wrong. A voice recording is personal data: turn this off if you would
# rather not retain it, and set a retention policy on media/audio/in/ if you do.
WHATSAPP_SAVE_INBOUND_AUDIO = config(
    "WHATSAPP_SAVE_INBOUND_AUDIO", default=True, cast=bool)

# Chat history settings (.pkl per phone number for follow-up questions)
WHATSAPP_CHAT_HISTORY_DIR = config(
    "WHATSAPP_CHAT_HISTORY_DIR", default=str(BASE_DIR / "chat_history"))
WHATSAPP_CHAT_HISTORY_MAX_TURNS = config(
    "WHATSAPP_CHAT_HISTORY_MAX_TURNS", default=25, cast=int)

# --- Generic WhatsApp API (/api/v1/whatsapp/message/) ----------------------
#
# The provider-independent endpoint carries no hook, so the catalogue it answers
# from has to come from somewhere. Left unset, a deployment with exactly one
# catalogue uses that one and nothing needs configuring; with several, the
# endpoint refuses rather than guess which corpus a farmer meant, and the caller
# can also name one per request.
# Cast handles the unset case itself: decouple applies it to the default too, so
# a `default=None` with an int cast raises at import.
WHATSAPP_DEFAULT_CATALOGUE_ID = config(
    "WHATSAPP_DEFAULT_CATALOGUE_ID", default="",
    cast=lambda v: int(v) if str(v or "").strip().isdigit() else None)

# AUTOMATIC ROUTING for that same endpoint: which catalogue answers which kind of
# question, so the farmer never has to name a corpus. "nutrient:5,variety:6,
# pest:7" — the categories are the ones hooks/routing.py knows how to recognise,
# and a category listed here is the only way it can be selected. Left unset,
# every generic message goes to WHATSAPP_DEFAULT_CATALOGUE_ID exactly as before,
# so routing is opt-in per deployment.
WHATSAPP_CATALOGUE_ROUTES = config("WHATSAPP_CATALOGUE_ROUTES", default="")

# Celery is NOT installed in this project. The async worker is the thread pool in
# hooks/service.py. Setting a broker here — after `pip install celery redis` and
# adding irri_ai/celery.py — is all that is needed for dispatch() to start using
# Celery instead; see hooks/tasks.py.
CELERY_BROKER_URL = config("CELERY_BROKER_URL", default="")

# --- Logging ---------------------------------------------------------------

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "django": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "irri": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
    },
}