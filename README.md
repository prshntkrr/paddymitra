# PaddyMitra AI — IRRI Knowledge Delivery System

A functional Django + MySQL + ChromaDB rebuild of the PaddyMitra AI prototype.

- **Backend:** Django 5 (class-based views, server-rendered templates)
- **Database:** MySQL (via XAMPP / phpMyAdmin)
- **Vector store:** ChromaDB (persistent, local) — one collection per Catalogue
- **AI pipeline:** OpenAI (text-embedding-3-small · gpt-4o-mini · whisper-1 · tts-1), translation + classification + RAG retrieval + TTS
- **WhatsApp gateway:** Picky Assist webhook + outbound sender
- **Chunking:** `langchain_text_splitters.RecursiveCharacterTextSplitter` (rule-based, open-source, deterministic)
- **Auth:** Django sessions, email-based login, 4 roles (Administrator / Editor / Knowledge Worker / Reviewer) with a `RoleRequiredMixin`.

---

## Prerequisites

1. **XAMPP** running (Apache optional; MySQL required). Default user `root` with empty password.
2. **Python 3.10+** on PATH.
3. **OpenAI API key** for the real AI pipeline (embeddings, LLM, STT, TTS).
4. **Picky Assist** credentials (optional — only needed to actually send/receive WhatsApp messages).

---

## Bring-up (first time)

```bash
# 1. Create the MySQL database (via phpMyAdmin OR CLI)
C:\xampp\mysql\bin\mysql.exe -u root -e "CREATE DATABASE irri_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 2. Virtualenv & deps
cd C:\xampp\htdocs\irri
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install "httpx<0.28"          # pin for openai 1.51 compatibility

# 3. Configure env
copy .env.example .env             # then edit .env and paste your OPENAI_API_KEY

# 4. Migrate
python manage.py migrate

# 5. Seed roles + demo users (password: qwerty@12)
python manage.py seed_roles_users

# 6. Import the 39-document classification CSV
python manage.py import_docs_csv "C:/Users/pc/Downloads/Doc_classification(Doc_sheet) (1).csv"

# 7. Seed sample catalogues, hooks, and ~50 fake queries (so dashboard is populated)
python manage.py seed_sample_data

# 8. Build the vector DB for one catalogue (real OpenAI embeddings; ~30 seconds)
python manage.py build_vector_db odisha-kharif-2026

# 9. Run the server
python manage.py runserver 8000
```

Open <http://127.0.0.1:8000/> and sign in with `admin@irri.local / qwerty@12`.

---

## Demo accounts (all password `qwerty@12`)

| Email | Role | Use for |
|---|---|---|
| `admin@irri.local` | Administrator | Everything (users, hooks, catalogues, CRUD) |
| `editor@irri.local` | Editor | Documents, keywords, catalogues CRUD |
| `kw@irri.local` | Knowledge Worker (KVK) | Upload documents → submit for review |
| `reviewer@irri.local` | Reviewer | Approve/reject the review queue |

---

## Key features to try

1. **Dashboard** — Highcharts monthly-trend, category distribution, district density, feedback KPIs, recent interactions. All driven by live ORM aggregates.
2. **Documents** — 39 imported documents with categories, subcategories, keywords, crop, content type, authors, journal. Filter by crop/content type, view detail, publish/unpublish.
3. **Approval workflow** — Log in as Knowledge Worker → upload a document → click "Submit for review" → log in as Reviewer → approve from `/documents/review/`.
4. **Keywords** — 101 auto-imported keywords with parent category, region, language, local names, expert validation; full CRUD.
5. **Catalogues** — Create a catalogue, attach documents, click **Build Vector DB** → Chroma collection built with real OpenAI embeddings.
6. **Catalogue search** — `/catalogues/<slug>/search/?q=...` runs embedding + similarity search and returns top-6 chunks with scores.
7. **Run demo query** — `/queries/demo/` fires the full pipeline: translate → classify → retrieve → generate → translate back (+ optional TTS). Every step is timed and recorded in `Query.pipeline_trace`.
8. **Hooks** — Configure WhatsApp trigger keywords → catalogue + language. Webhook endpoint: `POST /hooks/webhook/pickyassist/` (see `hooks/views.py` for payload shape).
9. **Users & roles** — `/accounts/users/` shows the permission matrix and all users with organization type.

---

## Management commands

```bash
python manage.py seed_roles_users
python manage.py import_docs_csv <path-to-csv>  [--reset]
python manage.py seed_sample_data  [--num-queries 50]
python manage.py build_vector_db <catalogue-slug>
```

---

## Architecture

```
irri_ai/        project package, settings split (base/dev/prod)
accounts/       custom email-based User + Role
documents/      Document, Category, Subcategory + CSV importer + approval workflow
keywords/       Keyword (parent -> subcategory -> region, translation + language, validation)
catalogues/     Catalogue (M2M Documents) + Chroma vector store services
hooks/          Hook (WhatsApp gateway binding) + Picky Assist webhook + outbound sender
queries/        Query + real AI pipeline service (pipeline.py) + demo form + reviewer flagging
dashboard/      DashboardView — ORM aggregates fed to Highcharts + Leaflet
core/           base.html, sidebar/topbar partials, RoleRequiredMixin, template tags, seed command
```

Services that do real work:

- `catalogues/services/chunking.py` — RecursiveCharacterTextSplitter + pypdf extraction with metadata fallback body.
- `catalogues/services/embeddings.py` — OpenAI `text-embedding-3-small` (with optional local MiniLM fallback when `USE_LOCAL_EMBEDDINGS=1`).
- `catalogues/services/vector_store.py` — ChromaDB PersistentClient, one collection per catalogue slug.
- `queries/services/pipeline.py` — end-to-end pipeline with tracing.
- `queries/services/whatsapp.py` — Picky Assist outbound.

---

## Why `RecursiveCharacterTextSplitter` for chunking

- **Open source** (MIT) and ships inside `langchain-text-splitters`.
- **Deterministic, zero-ML** — no model download needed; installs via `pip` on Windows.
- **Semantics-preserving** — splits on `\n\n` → `\n` → `. ` → ` ` → `` in that order, so paragraphs and sentences stay intact where possible.
- **Works on any text** — handles the messy mix of IRRI factsheets, research-paper abstracts, and keyword-heavy fallback bodies equally well.
- **Easy upgrade path** — can be swapped with `SemanticChunker` (from `langchain-experimental`) using the same interface when semantic chunking is required.

Configured in `settings/base.py`: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`.

---

## Known limitations / risks

- `build_vector_db` runs synchronously; large catalogues (~50+ docs with full PDFs) can take a few minutes. For production, move to django-q2 or Celery.
- Picky Assist webhook needs a publicly-reachable URL for real WhatsApp messages — use **ngrok** during local development.
- `USE_TZ = False` is set because XAMPP MySQL ships without MySQL's timezone tables (required by `TruncMonth` via `CONVERT_TZ`). Safe for single-region deployments.
- First query after server start downloads the embedding model metadata (a few seconds). Subsequent queries are hot.
- Some CSV documents lack an uploaded PDF — the ingestion pipeline falls back to a metadata-only body (`title + authors + journal + abstract + keywords + category + subcategory + countries`) so every one of the 39 tagged documents still produces useful embeddings.

---

## Troubleshooting

- **`mysqlclient` wheel fails to install on Windows** — the project has a PyMySQL shim in `irri_ai/__init__.py`. Just `pip install PyMySQL` and Django will use it transparently.
- **`TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`** — known `openai 1.51` vs `httpx>=0.28` clash. Fix: `pip install "httpx<0.28"`.
- **`Database returned an invalid datetime value`** — XAMPP MySQL missing timezone tables. We already set `USE_TZ=False` to sidestep this.
- **Chroma telemetry "capture() takes 1 positional argument but 3 were given"** — harmless posthog version mismatch; does not affect functionality.

---

## License

Internal prototype for IRRI / Indev Consultancy. Highcharts is free for non-commercial / personal use; a commercial license is required for commercial deployment.
