"""
Bulk ingest a folder of images + <prefix>.txt sidecars into the image bank
for a specific catalogue.

Usage:
    python manage.py ingest_image_folder <catalogue-slug> C:/path/to/folder
"""
from __future__ import annotations

from pathlib import Path

from django.core.files import File
from django.core.management.base import BaseCommand, CommandError

from catalogues.models import Catalogue
from image_bank.models import ImageGroup, KnowledgeImage
from image_bank.services.image_vector_store import add_image as vs_add
from image_bank.services.sidecar import IMG_EXT, find_sidecar


class Command(BaseCommand):
    help = "Ingest a folder of images + <prefix>.txt sidecars into a catalogue's image bank."

    def add_arguments(self, parser):
        parser.add_argument("catalogue_slug", help="Catalogue slug")
        parser.add_argument("folder", help="Path to folder containing images and sidecar .txt files")
        parser.add_argument("--skip-existing", action="store_true",
                            help="Skip images whose original_filename already exists for this catalogue.")

    def handle(self, *args, **options):
        try:
            catalogue = Catalogue.objects.get(slug=options["catalogue_slug"])
        except Catalogue.DoesNotExist:
            raise CommandError(f"Catalogue '{options['catalogue_slug']}' not found.")
        folder = Path(options["folder"]).resolve()
        if not folder.is_dir():
            raise CommandError(f"Not a directory: {folder}")

        skip_existing = options["skip_existing"]
        existing = set()
        if skip_existing:
            existing = set(
                KnowledgeImage.objects
                .filter(catalogue=catalogue)
                .values_list("original_filename", flat=True)
            )

        images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
        if not images:
            self.stdout.write(self.style.WARNING("No images found."))
            return

        search_dirs = [folder]

        created = 0
        indexed = 0
        skipped = 0
        errors = 0

        for img_path in images:
            if skip_existing and img_path.name in existing:
                skipped += 1
                continue
            try:
                prefix, description = find_sidecar(img_path, search_dirs=search_dirs)
                group, _ = ImageGroup.objects.update_or_create(
                    catalogue=catalogue, prefix=prefix,
                    defaults={"description": description},
                )
                with img_path.open("rb") as fh:
                    ki = KnowledgeImage(
                        group=group,
                        catalogue=catalogue,
                        prefix=prefix,
                        original_filename=img_path.name,
                        description=description,
                    )
                    ki.image.save(img_path.name, File(fh), save=True)
                created += 1
                try:
                    vs_add(ki)
                    indexed += 1
                except Exception as exc:
                    self.stderr.write(f"Index failed for {img_path.name}: {exc}")
                    errors += 1
            except Exception as exc:
                self.stderr.write(f"Failed {img_path}: {exc}")
                errors += 1

        self.stdout.write(self.style.SUCCESS(
            f"Ingested {created} images ({indexed} indexed) into '{catalogue.slug}'. "
            f"Skipped {skipped}, errors {errors}."
        ))
