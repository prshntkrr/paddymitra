"""
Evaluate retrieval quality across configurations.

Usage:
    python manage.py eval_retrieval --catalogue <slug>
    python manage.py eval_retrieval --catalogue <slug> --cross-encoder
    python manage.py eval_retrieval --catalogue <slug> --output results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

EVAL_QUERIES = [
    {"query": "How much urea for tillering?", "expects": "table", "topic": "dose"},
    {"query": "What is the nutrient content of poultry manure?", "expects": "table", "topic": "nutrient"},
    {"query": "Which varieties are recommended for upland?", "expects": "para", "topic": "variety"},
    {"query": "When to apply potash in rice?", "expects": "any", "topic": "stage"},
    {"query": "How to use organic manure?", "expects": "para", "topic": "organic"},
    {"query": "What is azolla nitrogen content?", "expects": "table", "topic": "nutrient"},
    {"query": "Seed rate for direct seeded rice", "expects": "any", "topic": "practice"},
    {"query": "Zinc deficiency symptoms in rice", "expects": "para", "topic": "deficiency"},
    {"query": "FYM application rate per hectare", "expects": "table", "topic": "dose"},
    {"query": "Combining organic and inorganic fertilizer", "expects": "para", "topic": "practice"},
]


class Command(BaseCommand):
    help = "Evaluate retrieval quality across configurations"

    def add_arguments(self, parser):
        parser.add_argument("--catalogue", required=True, help="Catalogue slug")
        parser.add_argument("--cross-encoder", action="store_true",
                            help="Enable cross-encoder for this run")
        parser.add_argument("--output", default="", help="Output JSON path")
        parser.add_argument("--n-results", type=int, default=6)

    def handle(self, **options):
        from catalogues.models import Catalogue
        from queries.services.pipeline import retrieve

        slug = options["catalogue"]
        try:
            catalogue = Catalogue.objects.get(slug=slug)
        except Catalogue.DoesNotExist:
            self.stderr.write(f"Catalogue '{slug}' not found.")
            return

        if options["cross_encoder"]:
            settings.USE_CROSS_ENCODER = True
            self.stdout.write("Cross-encoder ENABLED for this run.")

        n_results = options["n_results"]
        results = []

        for q in EVAL_QUERIES:
            t0 = time.perf_counter()
            hits = retrieve(catalogue, q["query"], n_results=n_results)
            elapsed = time.perf_counter() - t0

            content_types = {}
            for h in hits:
                ct = (h.get("metadata") or {}).get("content_type", "unknown")
                content_types[ct] = content_types.get(ct, 0) + 1

            has_table = any(
                (h.get("metadata") or {}).get("content_type") == "table"
                for h in hits
            )
            has_section = any(
                (h.get("metadata") or {}).get("section")
                for h in hits
            )

            result = {
                "query": q["query"],
                "topic": q["topic"],
                "expects_table": q["expects"] == "table",
                "got_table": has_table,
                "has_section_metadata": has_section,
                "n_hits": len(hits),
                "content_types": content_types,
                "latency_ms": round(elapsed * 1000),
                "top_chunk_preview": (hits[0].get("chunk") or "")[:150] if hits else "",
            }
            results.append(result)

            status = "OK" if (q["expects"] != "table" or has_table) else "MISS"
            self.stdout.write(
                f"[{status}] {q['query'][:50]:50s} "
                f"hits={len(hits)} table={has_table} "
                f"latency={result['latency_ms']}ms "
                f"types={content_types}"
            )

        # Summary
        total = len(results)
        table_queries = [r for r in results if r["expects_table"]]
        table_hit_rate = (
            sum(1 for r in table_queries if r["got_table"]) / len(table_queries)
            if table_queries else 0
        )
        section_rate = sum(1 for r in results if r["has_section_metadata"]) / total
        avg_latency = sum(r["latency_ms"] for r in results) / total

        self.stdout.write("\n--- Summary ---")
        self.stdout.write(f"Queries evaluated:     {total}")
        self.stdout.write(f"Table hit rate:        {table_hit_rate:.0%} ({len(table_queries)} dose/nutrient queries)")
        self.stdout.write(f"Section metadata rate: {section_rate:.0%}")
        self.stdout.write(f"Avg latency:           {avg_latency:.0f}ms")
        self.stdout.write(f"Cross-encoder:         {'ON' if getattr(settings, 'USE_CROSS_ENCODER', False) else 'OFF'}")

        if options["output"]:
            out_path = Path(options["output"])
            out_path.write_text(
                json.dumps({
                    "config": {
                        "catalogue": slug,
                        "n_results": n_results,
                        "cross_encoder": getattr(settings, "USE_CROSS_ENCODER", False),
                        "rrf_k": getattr(settings, "RRF_K", 60),
                        "sherpa_url": getattr(settings, "LLM_SHERPA_API_URL", ""),
                    },
                    "summary": {
                        "table_hit_rate": table_hit_rate,
                        "section_metadata_rate": section_rate,
                        "avg_latency_ms": avg_latency,
                    },
                    "results": results,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.stdout.write(f"Results saved to: {out_path}")
