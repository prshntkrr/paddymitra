import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "irri_ai.settings.dev")
django.setup()

from queries.services.entity_intent import analyse, variety_conditions, named_varieties
from queries.services.pipeline import _variety_input_note, retrieve_across_catalogues

# Simulate the actual pipeline: multi-catalogue retrieval
hits = retrieve_across_catalogues("What is the fertilizer dose for MTU 1010?", [7, 5], n_results=10)
print("--- multi-catalogue hits ---")
for i, h in enumerate(hits):
    title = (h.get("metadata") or {}).get("document_title", "?")
    print(i, title[:40], "|", h["chunk"][:100])

intent = analyse("What is the fertilizer dose for MTU 1010?")
print("\nconditions from multi-cat hits:", variety_conditions("MTU 1010", hits))
note = _variety_input_note(intent, hits)
print("note:", note[:300] if note else "EMPTY")
