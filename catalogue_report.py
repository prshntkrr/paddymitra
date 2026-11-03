"""Where each question's corpus comes from. Read-only.

Save at the project root (beside manage.py) and run:
    python catalogue_report.py
"""
import os
import sys

import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "irri_ai.settings.prod")
django.setup()

from django.conf import settings                                    # noqa: E402
from catalogues.models import Catalogue                             # noqa: E402
from catalogues.services.vector_store import get_client             # noqa: E402
from hooks.models import Hook                                       # noqa: E402
from queries.models import FAQ                                      # noqa: E402

counts = {}
try:
    counts = {c.name: c.count() for c in get_client().list_collections()}
except Exception as exc:                                            # noqa: BLE001
    print("chroma unreadable:", exc)

print("\nCATALOGUES")
for c in Catalogue.objects.all().order_by("pk"):
    name = c.vector_store_collection_name or c.slug
    n = counts.get(name)
    if n is None:
        state = "NO COLLECTION - build_vector_db " + c.slug
    elif n == 0:
        state = "EMPTY - build_vector_db " + c.slug
    else:
        state = "%d chunks" % n
    print("  id=%-4s %-28s %-10s %s" % (c.pk, c.slug, c.status, state))

print("\nHOOKS (each answers from exactly one catalogue)")
for h in Hook.objects.all().order_by("pk"):
    cat = h.catalogue
    where = ("catalogue id=%s %s" % (cat.pk, cat.slug)) if cat else "NO CATALOGUE SET"
    print("  %-10s %-8s -> %s" % (h.trigger_keyword, h.status, where))
if not Hook.objects.filter(status=Hook.STATUS_ACTIVE).exists():
    print("  (no active hook: every WhatsApp message is recorded and ignored)")

reachable = set(Hook.objects.filter(status=Hook.STATUS_ACTIVE)
                .values_list("catalogue_id", flat=True))
unreachable = [c for c in Catalogue.objects.all() if c.pk not in reachable]
if unreachable:
    print("  not reachable from WhatsApp (no active hook points at them):")
    for c in unreachable:
        print("     id=%s %s" % (c.pk, c.slug))

print("\nGENERIC API default")
cid = getattr(settings, "WHATSAPP_DEFAULT_CATALOGUE_ID", None)
if cid:
    ok = Catalogue.objects.filter(pk=cid).first()
    print("  WHATSAPP_DEFAULT_CATALOGUE_ID=%s -> %s"
          % (cid, ok.slug if ok else "DOES NOT EXIST - every generic call 503s"))
else:
    n = Catalogue.objects.count()
    print("  unset, and %d catalogue(s) exist -> %s"
          % (n, "fine, the only one is used" if n == 1
             else "generic calls 503 unless the request names one"))

print("\nAUTOMATIC ROUTING (generic endpoint, WHATSAPP_CATALOGUE_ROUTES)")
try:
    from hooks.routing import routes as _routes
    table = _routes()
except Exception as exc:                                            # noqa: BLE001
    table, exc_text = {}, str(exc)
    print("  router unavailable: %s" % exc_text)
if not table:
    print("  no routes configured - every generic message uses the default")
for category, cid in sorted(table.items()):
    cat = Catalogue.objects.filter(pk=cid).first()
    if cat is None:
        print("  %-9s -> id=%-4s MISSING: this category silently falls back "
              "to the default" % (category, cid))
    elif cid == getattr(settings, "WHATSAPP_DEFAULT_CATALOGUE_ID", None):
        print("  %-9s -> id=%-4s %s  (SAME as the default - routing this "
              "category changes nothing)" % (category, cid, cat.slug))
    else:
        print("  %-9s -> id=%-4s %s" % (category, cid, cat.slug))

print("\nFAQ entries answering for EVERY catalogue: %d"
      % FAQ.objects.filter(is_active=True, catalogue__isnull=True).count())
