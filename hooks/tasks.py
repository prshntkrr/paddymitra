"""
Celery tasks for WhatsApp message answering.

CELERY IS NOT INSTALLED IN THIS PROJECT TODAY.
---------------------------------------------
There is no `celery` in requirements.txt, no celery.py, no broker setting. The
async processing that exists is the bounded thread pool in hooks/service.py, and
that is what dispatch() uses.

This module is written so that the day Celery arrives, nothing here has to
change: the import is guarded, `answer_message_task` is None when Celery is
absent, and `service.celery_available()` checks for both the task AND a
configured broker before handing it any work. Until then it is inert.

TO TURN IT ON
-------------
    pip install celery redis
    # irri_ai/settings/base.py
    CELERY_BROKER_URL = config("CELERY_BROKER_URL", default="redis://localhost:6379/0")
    # irri_ai/celery.py — the standard app, autodiscovering tasks
    celery -A irri_ai worker -l info

`dispatch()` starts routing through Celery on its own once the broker is set.

WHY THE TASK IS THIN
--------------------
Everything it does lives in service.answer_message(), which already records its
own failures on the InboundMessage row. The task therefore has nothing to catch
and nothing to decide: a worker that dies leaves the row in `processing`, and
`manage.py whatsapp_retry` is what finishes it either way. That is the same
recovery story as the thread pool, which is what keeps the two dispatch paths
behaving identically.
"""
from __future__ import annotations

import logging

log = logging.getLogger("irri")

try:                                                 # pragma: no cover
    from celery import shared_task
except ImportError:
    shared_task = None

answer_message_task = None

if shared_task is not None:                          # pragma: no cover
    # Retries are for TRANSPORT failures only — a task that never started. A
    # failure inside answer_message() is already recorded and must not be retried
    # blindly here, because the expensive half may have succeeded: the retry
    # would either pay for the answer twice or send it twice. answer_message()
    # itself is idempotent about that, and whatsapp_retry is the deliberate,
    # rate-limited way to pick work back up.
    @shared_task(name="hooks.answer_whatsapp_message", bind=True,
                 max_retries=2, default_retry_delay=30, acks_late=True)
    def answer_message_task(self, inbound_id: int) -> str:
        from .service import answer_message
        return answer_message(inbound_id)
