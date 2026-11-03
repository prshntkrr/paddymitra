# PaddyMitra — Production Deploy Runbook (2026-11 content + pipeline session)

Brings **paddymitra-dev.indevconsultancy.in** (DB `paddymitra_dev_db`) in line with
the local changes from this session. Work top-to-bottom. Keep the backups until
the verification step passes.

> Placeholders: `<db_user>` = production MySQL user (from the prod `.env`
> `DATABASES['default']['USER']`); `<PROD>` = the project directory on the server
> (the folder that contains `manage.py`); `<prod-venv>` = the production Python.
> Run every `manage.py` command from `<PROD>` with the production settings module.

---

## 0. What this deploys

**Code (7 files — the project folder is not a git repo here, so copy them):**
| File | Why |
|---|---|
| `queries/services/pipeline.py` | answer fixes (see §6) |
| `queries/services/entity_intent.py` | variety-list detection tolerates the "varities"/"varity" misspelling; months no longer read as variety names |
| `queries/services/faq.py` | FAQ guard: a "how much" (quantity) question no longer matches a "when" (timing) FAQ |
| `catalogues/services/chunking.py` | reads the content-hash cache for **any** file type (xlsx/csv/md), not just PDF |
| `irri_ai/settings/base.py` | adds `USE_DOCLING` + `DOCLING_CACHE_DIR` settings |
| `queries/management/commands/sync_faq_vector.py` | new command — FAQ → vector document |
| `catalogues/management/commands/migrate_session_content.py` | new command — creates the docs + FAQs |

**Data:**
| Item | Source (local) | Destination (prod) |
|---|---|---|
| Cache (84 JSON, 2.4 MB) | `docling_cache/*.json` | `<PROD>/docling_cache/` |
| FAQ fixture (6 MB) | `deploy/faq_fixture.json` | `<PROD>/deploy/faq_fixture.json` |
| Variety dataset | `media/documents/Odisha_Paddy_varieties_filled.xlsx` | same path under `<PROD>` |
| Dose reference | `media/documents/Rice-Fertilizer-Manure-Dose-Quick-Reference.md` | same |
| Nutrient advisories PDF | `media/documents/All varieties NUTRIENT ADVISORIES.pdf` | same |

**Not copied / not created:** `chroma_db/`, the local DB, and documents 87/88
(the two Q&A sheets — their content lives in the 192 FAQs and in the
`sync_faq_vector` document, so they are intentionally left out).

> **Why the vector store is rebuilt on prod, never copied.** Each vector's id is
> `{document_id}:{chunk_index}` and its metadata carries `document_id` — these are
> the **local** Django primary keys (see `catalogues/services/vector_store.py`).
> Prod assigns its own PKs when `migrate_session_content` creates the docs, so a
> copied local store (≈170 MB) would point every chunk at the wrong prod row and
> break retrieval silently. Re-embedding the ~1,900 chunks on prod via
> `build_vector_db` (§5) costs a few cents and a few minutes, and is the only
> correct way to get the embeddings onto the server.

---

## 1. Pre-flight (on the server)

```bash
cd <PROD>
<prod-venv> manage.py shell -c "from django.conf import settings; print('DB', settings.DATABASES['default']['NAME']); print('USE_DOCLING', getattr(settings,'USE_DOCLING', 'DEFAULT-TRUE')); print('CACHE', getattr(settings,'DOCLING_CACHE_DIR','DEFAULT'))"
```
Confirm the DB is `paddymitra_dev_db`. `USE_DOCLING` must be true (it defaults to
true once `settings/base.py` is deployed; make sure the prod `.env` does **not**
set it to false).

---

## 2. Back up first (do not skip)

```bash
mysqldump -u <db_user> -p paddymitra_dev_db | gzip > ~/paddymitra_dev_db_$(date +%F_%H%M).sql.gz
```
```bash
cd <PROD> && tar czf ~/chroma_db_$(date +%F_%H%M).tgz chroma_db/
```

---

## 3. Deploy the 5 code files

Copy each file from local to the same relative path under `<PROD>` (scp/rsync or
your pipeline). Example with rsync from the local machine:

```bash
rsync -av queries/services/pipeline.py queries/services/entity_intent.py queries/services/faq.py catalogues/services/chunking.py irri_ai/settings/base.py queries/management/commands/sync_faq_vector.py catalogues/management/commands/migrate_session_content.py <user>@paddymitra-dev.indevconsultancy.in:<PROD>/<same-relative-paths>
```
(Keep the directory structure — e.g. `pipeline.py` must land in `<PROD>/queries/services/`.)

---

## 4. Copy the data

```bash
# from the local project dir
rsync -av docling_cache/*.json  <user>@HOST:<PROD>/docling_cache/
rsync -av deploy/faq_fixture.json <user>@HOST:<PROD>/deploy/
rsync -av "media/documents/Odisha_Paddy_varieties_filled.xlsx" \
          "media/documents/Rice-Fertilizer-Manure-Dose-Quick-Reference.md" \
          "media/documents/All varieties NUTRIENT ADVISORIES.pdf" \
          <user>@HOST:<PROD>/media/documents/
```

---

## 5. Create the documents + FAQs, then build the vector DB

Run **in this order** on the server, from `<PROD>`:

```bash
<prod-venv> manage.py migrate_session_content
```
Creates docs 86/89/90 (prod assigns its own IDs — fine), attaches them, and loads
the 192 FAQ rows (create-if-missing; it never overwrites a production FAQ). Expect
"3 created … 192 created".

```bash
<prod-venv> manage.py sync_faq_vector --replace-sheets
```
Builds the "Rice Knowledge Base FAQ (all entries)" document (192 FAQ chunks) and
detaches the sheet docs 87/88 **if they exist on prod** (harmless if they don't).

```bash
<prod-venv> manage.py build_vector_db rice-knowledge-base
```
Re-embeds the whole catalogue via OpenAI (needs `OPENAI_API_KEY` set on prod).
Expect roughly **~1,900 chunks** across the catalogue's documents.

---

## 6. Restart the app

Restart however prod runs it, e.g. one of:
```bash
sudo systemctl restart gunicorn      # or: supervisorctl restart paddymitra
# passenger/mod_wsgi: touch <PROD>/tmp/restart.txt
```
This is required — `pipeline.py` is loaded once per worker. The fixes:
1. **fertiliser → products / nutrient → elements** in answers
2. **stage-dose with no area** ("how much urea at tillering") no longer refuses
3. **ha↔acre conversion** ("80 kg/ha for 2 acres" → ~65 kg, not 160)
4. **bare per-acre product dose** ("how much urea per acre") no longer refuses
5. **variety fertiliser** gives the duration advisory instead of always asking land type
6. **variety-list typo tolerance** ("name some 120 days **varities**") still lists the varieties instead of refusing (needs `entity_intent.py` too)
7. **"how much `<stage>` fertilizer"** returns the grounded stage dose, never a bare timing sentence ("At 20–25 days after transplanting") — the real fix is the FAQ quantity-vs-timing guard in `faq.py` (a "when" FAQ no longer answers a "how much" question); the `pipeline.py` dash-fix is a secondary safety
8. **no month-dates in variety lists** — "July 15"/"Aug 30" sowing-window dates no longer appear as variety names (`entity_intent.py`)

None of these need a vector rebuild — all are query-time logic. (The rebuild in §5 is for the content/FAQ documents, not these fixes.)

---

## 7. Verify

```bash
<prod-venv> manage.py shell -c "
from catalogues.models import Catalogue
from queries.services.pipeline import retrieve, generate_answer
c=Catalogue.objects.get(slug='rice-knowledge-base')
for q in ['How much urea per acre?','What fertiliser is used for Pooja Dhan?','If the fertilizer rate is 80 kg/ha, how much for 2 acres?','How many days old should rice seedlings be for transplanting?','name some 120 days varities','How much tillering stage fertilizer should I give to the rice?']:
    print('Q:',q); print('A:',generate_answer(q,retrieve(c,q,n_results=8))[:200]); print()
"
```
Expect: urea-per-acre → a schedule (not a refusal); Pooja → DAP/Urea/MOP schedule
(no land-type question); 80 kg/ha → ~65 kg; seedling age → 20–25 days; "120 days
varities" (typo) → a list of medium-duration varieties (not a refusal); tillering
fertilizer → "30–35 kg Urea per acre" (a dose, not a timing sentence).

Also sanity-check counts:
```bash
<prod-venv> manage.py shell -c "from queries.models import FAQ; from catalogues.models import Catalogue; c=Catalogue.objects.get(slug='rice-knowledge-base'); print('docs',c.documents.count(),'faqs',FAQ.objects.filter(is_active=True).count())"
```

---

## 8. Rollback (if verification fails)

```bash
# DB
gunzip -c ~/paddymitra_dev_db_<ts>.sql.gz | mysql -u <db_user> -p paddymitra_dev_db
# vector store
cd <PROD> && rm -rf chroma_db && tar xzf ~/chroma_db_<ts>.tgz
# code: restore the 5 files from your previous copies / VCS, then restart the app
```

---

## Notes & caveats

- **Existing-doc clean renderings.** The cache also holds the clean-rendered
  chunks for the 8 variety factsheets, the STRVs catalogue, and the submergence
  bulletin tables (from earlier this session). Those apply on prod **only if the
  matching source files already exist on prod with identical bytes** (the cache
  key is the file's SHA-256). Where they differ or are absent, those docs fall
  back to their normal extraction — no harm, just no improvement. The new docs
  (86/89/90/91) are fully handled here because their files are copied.
- **Re-running** any of the three commands is safe (idempotent).
- **`migrate_session_content`** skips FAQs that already exist by normalised
  question; pass `--force-update-faqs` only if you intend to overwrite.
- **Not deployed to prod yet** when this runbook was written — these changes are
  all local. Nothing here has been run against production.
