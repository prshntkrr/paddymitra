"""
Build (or rebuild) a catalogue's Chroma vector database.
"""
from django.core.management.base import BaseCommand, CommandError

from catalogues.models import Catalogue
from catalogues.services.vector_store import build_catalogue_vector_db


class Command(BaseCommand):
    help = "Build the Chroma vector database for a catalogue."

    def add_arguments(self, parser):
        parser.add_argument("slug", help="Catalogue slug")

    def handle(self, *args, **options):
        slug = options["slug"]
        try:
            catalogue = Catalogue.objects.get(slug=slug)
        except Catalogue.DoesNotExist:
            raise CommandError(f"Catalogue with slug '{slug}' not found.")
        self.stdout.write(f"Building vector DB for '{catalogue.name}'...")
        stats = build_catalogue_vector_db(catalogue)
        self.stdout.write(self.style.SUCCESS(
            f"Done: {stats['documents']} documents -> {stats['chunks']} chunks."
        ))
