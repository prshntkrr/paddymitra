"""Which hook (if any) answered the last messages, and from which catalogue.

Save beside manage.py and run:  python hook_trace.py
Read-only.
"""
import os
import sys

import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "irri_ai.settings.prod")
django.setup()

from hooks.models import InboundMessage                             # noqa: E402

rows = (InboundMessage.objects.select_related("hook", "query", "query__catalogue")
        .order_by("-created_at")[:15])
if not rows:
    print("no inbound messages recorded yet")

for m in rows:
    hook = m.hook.trigger_keyword if m.hook else "NO HOOK (generic path)"
    q = m.query
    cat = q.catalogue.name if (q and q.catalogue) else "-"
    chunks = len((q.pipeline_trace or {}).get("retrieved_chunks") or []) if q else 0
    print("%s  %-9s provider=%-8s hook=%-22s catalogue=%-28s chunks=%s"
          % (m.created_at.strftime("%d %b %H:%M"), m.status, m.provider,
             hook, cat, chunks))
    print("      text: %r" % ((m.text or "")[:70],))
    keys = sorted((m.raw_payload or {}).keys())
    print("      payload keys: %s" % (", ".join(keys) or "(none)",))
