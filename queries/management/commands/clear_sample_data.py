"""
Remove seeded dummy data (fake queries + sample hooks + sample catalogues).

Real data preserved:
  - Users, Roles
  - Documents, Categories, Subcategories, Keywords (all imported from CSV)
  - Queries that came from the real pipeline (have populated pipeline_trace.steps)
"""
from django.core.management.base import BaseCommand

from catalogues.models import Catalogue
from hooks.models import Hook
from queries.models import Query


SAMPLE_CATALOGUE_NAMES = {
    "Odisha Kharif 2026",
    "Rice Disease Catalogue",
    "Fertiliser Advisory 2026",
}
SAMPLE_HOOK_KEYWORDS = {"DHAN", "ROGA", "SAAR"}


class Command(BaseCommand):
    help = "Clear sample dummy data (seeded queries/hooks/catalogues). Keeps real content."

    def add_arguments(self, parser):
        parser.add_argument("--keep-catalogues", action="store_true",
                            help="Keep the sample catalogues (delete only queries + hooks).")
        parser.add_argument("--all-queries", action="store_true",
                            help="Delete ALL queries (including real pipeline runs).")

    def handle(self, *args, **options):
        # 1. Delete seeded queries. Real pipeline runs have pipeline_trace.steps.
        if options["all_queries"]:
            n = Query.objects.all().count()
            Query.objects.all().delete()
            self.stdout.write(self.style.WARNING(f"Deleted ALL {n} queries (real + seeded)."))
        else:
            qs = Query.objects.all()
            seeded = [q for q in qs if not (q.pipeline_trace or {}).get("steps")]
            for q in seeded:
                q.delete()
            self.stdout.write(self.style.SUCCESS(
                f"Deleted {len(seeded)} seeded queries. "
                f"Real pipeline queries remaining: {Query.objects.count()}"
            ))

        # 2. Delete sample hooks
        deleted_hooks, _ = Hook.objects.filter(trigger_keyword__in=SAMPLE_HOOK_KEYWORDS).delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted_hooks} sample hooks."))

        # 3. Optionally delete sample catalogues (but keep any that have extra hooks or are referenced)
        if not options["keep_catalogues"]:
            for name in SAMPLE_CATALOGUE_NAMES:
                c = Catalogue.objects.filter(name=name).first()
                if c and not c.hooks.exists():
                    # detach documents (M2M only — documents themselves preserved) and delete
                    c.documents.clear()
                    c.delete()
                    self.stdout.write(f"  Deleted sample catalogue: {name}")
                elif c:
                    self.stdout.write(self.style.WARNING(
                        f"  Kept catalogue '{name}' because it still has linked hooks."
                    ))
        self.stdout.write(self.style.SUCCESS("Done. You can now create your own catalogues / hooks / queries."))
