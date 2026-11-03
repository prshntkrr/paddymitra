"""
Create one real catalogue containing every published document and build its vector DB.
Gives you a working RAG corpus out of the box — no dummy data.
"""
from django.core.management.base import BaseCommand

from catalogues.models import Catalogue
from catalogues.services.vector_store import build_catalogue_vector_db
from documents.models import Document


class Command(BaseCommand):
    help = "Create a real catalogue from all Published documents and build its vector DB."

    def add_arguments(self, parser):
        parser.add_argument("--name", default="Rice Knowledge Base")
        parser.add_argument("--slug", default="rice-knowledge-base")
        parser.add_argument("--skip-build", action="store_true",
                            help="Skip vector DB build (useful if OPENAI key unavailable).")

    def handle(self, *args, **options):
        docs = Document.objects.filter(status="Published")
        if not docs.exists():
            self.stdout.write(self.style.ERROR(
                "No published documents found. Run import_docs_csv first."
            ))
            return

        catalogue, created = Catalogue.objects.get_or_create(
            slug=options["slug"],
            defaults={
                "name": options["name"],
                "description": "All published agricultural documents imported from source.",
                "geography": "All India",
                "season_year": "All Year",
                "status": Catalogue.STATUS_PUBLISHED,
            },
        )
        catalogue.documents.set(docs)
        catalogue.save()
        verb = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(
            f"{verb} catalogue '{catalogue.name}' with {docs.count()} documents."
        ))

        if options["skip_build"]:
            self.stdout.write("Skipping vector DB build.")
            return

        self.stdout.write("Building vector DB (real OpenAI embeddings)...")
        stats = build_catalogue_vector_db(catalogue)
        self.stdout.write(self.style.SUCCESS(
            f"Vector DB ready: {stats['chunks']} chunks from {stats['documents']} documents."
        ))
