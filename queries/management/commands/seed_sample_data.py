"""
Create sample catalogues, hooks, and fake queries so the dashboard is not empty.
"""
import random
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from catalogues.models import Catalogue
from documents.models import Document
from hooks.models import Hook
from queries.models import Query


ODISHA_DISTRICTS = [
    "Bargarh", "Puri", "Cuttack", "Ganjam", "Khordha", "Sambalpur", "Balasore",
    "Bhadrak", "Dhenkanal", "Jajpur", "Kendrapara", "Mayurbhanj", "Nayagarh",
    "Nuapada", "Rayagada", "Sonepur", "Sundargarh",
]
CATEGORIES = ["Rice Variety", "Disease", "Fertiliser", "Water Mgmt", "Others"]
TASK_TYPES = ["Diagnosis", "Variety Suitability", "Fertiliser Rec.", "Management Advice"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
CROP_STAGES = ["Nursery", "Tillering", "Flowering", "Mature Grain"]
FEEDBACK_DIST = ["Good"] * 6 + ["Average"] * 3 + ["Bad"]
PROBLEM_ENTITIES = [
    "Blast", "BPH", "Urea dose", "Swarna variety", "Water depth",
    "MTU-1010", "Leaf blight", "DAP", "Transplanting", "Lalat",
]
SAMPLE_QUERIES = [
    "What is the treatment for blast disease in rice?",
    "Is Swarna rice suitable for Kharif season in Puri?",
    "How much urea should be applied per acre for paddy?",
    "How to control BPH pest in paddy field?",
    "Tell me about MTU-1010 rice variety features",
    "When to apply DAP fertiliser in rice crop?",
    "What is the recommended water depth for transplanted rice?",
    "Which rice variety is best for waterlogged areas?",
    "How to diagnose leaf blight in my field?",
    "Can I apply micronutrients along with urea?",
]


class Command(BaseCommand):
    help = "Seed sample catalogues, hooks, and ~50 fake queries for demo."

    def add_arguments(self, parser):
        parser.add_argument("--num-queries", type=int, default=50)

    def handle(self, *args, **options):
        n = options["num_queries"]
        docs = list(Document.objects.filter(status="Published"))
        if not docs:
            self.stdout.write(self.style.WARNING(
                "No published documents found. Run import_docs_csv first."
            ))
            return

        # Catalogues
        catalogue_specs = [
            ("Odisha Kharif 2026", "Rice advisories for Odisha Kharif season", "Odisha", "Kharif 2026"),
            ("Rice Disease Catalogue", "Pest and disease knowledge for rice", "All India", "All Year"),
            ("Fertiliser Advisory 2026", "Nutrient & fertiliser recommendations", "All India", "All Year"),
        ]
        catalogues = []
        for name, desc, geo, season in catalogue_specs:
            c, created = Catalogue.objects.get_or_create(
                name=name,
                defaults={
                    "description": desc, "geography": geo, "season_year": season,
                    "status": Catalogue.STATUS_PUBLISHED,
                },
            )
            # attach a subset of documents
            sample = random.sample(docs, min(len(docs), random.randint(8, 20)))
            c.documents.set(sample)
            catalogues.append(c)
            self.stdout.write(("created" if created else "exists") + f": Catalogue '{c.name}' ({c.documents.count()} docs)")

        # Hooks
        hook_specs = [
            ("Odisha Kharif Advisory", "+91 74000 00001", "DHAN", catalogues[0], "Odia"),
            ("Rice Disease Hook", "+91 74000 00002", "ROGA", catalogues[1], "Odia"),
            ("Fertiliser Hook", "+91 74000 00003", "SAAR", catalogues[2], "Hindi"),
        ]
        for name, wapp, keyword, catalogue, lang in hook_specs:
            h, _ = Hook.objects.update_or_create(
                trigger_keyword=keyword.upper(),
                gateway_provider=Hook.GATEWAY_PICKY,
                defaults={
                    "name": name, "whatsapp_number": wapp,
                    "catalogue": catalogue, "primary_language": lang,
                    "status": Hook.STATUS_ACTIVE, "messages_processed_count": random.randint(200, 4000),
                    "avg_response_time": round(random.uniform(4.5, 7.5), 2),
                    "success_rate": round(random.uniform(90, 99), 1),
                    "webhook_url": "/hooks/webhook/pickyassist/",
                },
            )

        # Sample queries
        now = timezone.now()
        for i in range(n):
            ts = now - timedelta(days=random.randint(0, 29), hours=random.randint(0, 23),
                                 minutes=random.randint(0, 59))
            hook = random.choice(list(Hook.objects.all()))
            cat = hook.catalogue
            q_text = random.choice(SAMPLE_QUERIES)
            category = random.choice(CATEGORIES)
            q = Query.objects.create(
                whatsapp_number="+91 94XXX " + str(10000 + i),
                district=random.choice(ODISHA_DISTRICTS),
                hook=hook,
                trigger_keyword=hook.trigger_keyword,
                catalogue=cat,
                category=category,
                task_type=random.choice(TASK_TYPES),
                problem_entity=random.choice(PROBLEM_ENTITIES),
                season=random.choice(SEASONS),
                crop_stage=random.choice(CROP_STAGES),
                missing_context=random.choice(["", "", "Missing: variety", "Missing: crop stage, variety"]),
                original_query_language=hook.primary_language,
                original_query_text=q_text,
                translated_query_text=q_text,
                ai_response_text="Sample response — for demo purposes. Apply "
                                  "Tricyclazole 75% WP @ 0.6 g/litre at first symptoms.",
                ai_response_text_local="ନମୁନା ଉତ୍ତର ...",
                response_time_seconds=round(random.uniform(3.0, 8.5), 2),
                farmer_feedback=random.choice(FEEDBACK_DIST),
                status="Responded",
            )
            q.timestamp = ts
            q.save(update_fields=["timestamp"])
            # attach 1-3 source docs
            q.source_documents.set(random.sample(list(cat.documents.all()) or docs[:5],
                                                 min(3, cat.documents.count() or 3)))

        self.stdout.write(self.style.SUCCESS(
            f"Seeded {len(catalogues)} catalogues, {Hook.objects.count()} hooks, {n} sample queries."
        ))
