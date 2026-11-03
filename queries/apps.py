from django.apps import AppConfig
import logging
import os
import sys
import threading

from django.conf import settings


class QueriesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "queries"

    def ready(self):
        """Preload the local Odia Wav2Vec2 model in background on server start.

        This triggers the model download if it's not already present in the
        Hugging Face cache. Runs in a daemon thread so it doesn't block startup.

        OFF BY DEFAULT, because it is a FALLBACK that costs ~1.2 GB of resident
        memory whether or not it is ever used. Sarvam answers every Odia
        transcription while it is reachable; the local model only runs when
        Sarvam fails, and _get_local_pipe() loads it lazily at that moment. So
        preloading buys a little latency on a rare path and pays for it with a
        permanent memory reservation — a bad trade on a small server, and the
        reason a dev server on a 8 GB machine was being killed after a few
        minutes. Set PRELOAD_LOCAL_STT=True where there is memory to spare.
        """
        log = logging.getLogger("irri")

        # THE AUTORELOADER RUNS ready() TWICE. `runserver` starts a parent that
        # watches files and a child that serves; ready() fires in both, so an
        # unguarded preload loaded the model TWICE and doubled the cost. Only
        # the child has RUN_MAIN set. Guarded narrowly so that gunicorn, celery
        # and management commands — none of which set RUN_MAIN — are untouched.
        is_reloader_parent = ("runserver" in sys.argv
                              and os.environ.get("RUN_MAIN") != "true")
        if is_reloader_parent:
            return

        if not getattr(settings, "PRELOAD_LOCAL_STT", False):
            log.info("local Odia STT preload is off (PRELOAD_LOCAL_STT); the "
                     "model will load on first use if Sarvam is unavailable")
            return

        def _preload():
            try:
                # Lazy import — this will download the model if needed.
                from .services import indic_stt, pipeline

                log.info("Preloading local Odia Wav2Vec2 model in background...")
                try:
                    indic_stt._get_local_pipe()
                    log.info("Local Odia Wav2Vec2 model ready.")
                    # log.info("Preloading MMS Odia TTS...")
                    # mms_tts.get_model()
                    # log.info("MMS Odia TTS ready.")
                except Exception as exc:
                    log.warning("Failed to preload Wav2Vec2 model: %s", exc)

            except Exception as exc:
                log.warning("Failed to start model preload thread: %s", exc)

        thread = threading.Thread(target=_preload, name="wav2vec-preload", daemon=True)
        thread.start()
