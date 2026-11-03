#!/usr/bin/env bash
#
# PaddyMitra — server-side deploy commands
# =========================================
# Run on the server (e.g. paddymitra-dev.indevconsultancy.in) from the project
# directory (the folder that contains manage.py), AFTER the code is pulled.
#
#   1) Edit the two variables below.
#   2) bash deploy/run_on_server.sh
#
# The vector DB is BUILT HERE, on the server. Do NOT copy a local chroma_db:
# every vector id is "{document_id}:{chunk_index}" and its metadata carries the
# Django document PK, so a store built on another machine points at the wrong
# rows. Building here makes the ids match this server's document rows.
#
set -euo pipefail

# ----------------------------- EDIT THESE TWO -----------------------------
PROD_DIR="/var/www/paddymitra"              # folder that contains manage.py
VENV_PY="/var/www/paddymitra/venv/bin/python"   # the production Python (venv)
# --------------------------------------------------------------------------

cd "$PROD_DIR"
STAMP="$(date +%F_%H%M)"

echo "== 0. Sanity check =="
"$VENV_PY" manage.py shell -c "from django.conf import settings; print('DB     ', settings.DATABASES['default']['NAME']); print('CHROMA ', settings.CHROMA_PERSIST_DIR)"
# Expect DB = paddymitra_dev_db. Make sure OPENAI_API_KEY is set in the prod .env
# (build_vector_db re-embeds via OpenAI and will fail without it).

echo "== 1. Pull the new code =="
git fetch origin
git checkout main
git pull --ff-only origin main

echo "== 2. Media for the NEW documents (these are NOT in git) =="
# media/ is gitignored, so the 3 source files for the new documents must already
# be present on the server (copy them with scp/rsync if missing). Without them,
# the build produces no chunks for those docs.
for f in \
  "media/documents/Odisha_Paddy_varieties_filled.xlsx" \
  "media/documents/Rice-Fertilizer-Manure-Dose-Quick-Reference.md" \
  "media/documents/All varieties NUTRIENT ADVISORIES.pdf"; do
  if [ ! -f "$f" ]; then
    echo "!! MISSING: $f  — copy it to media/documents/ before continuing."; exit 1
  fi
done
echo "   all 3 media docs present."

echo "== 3. Back up the current DB + vector store (rollback safety) =="
# This keeps the OLD vector DB as a tarball before we build the new one, so you
# can always roll back. Uncomment the mysqldump line and fill in the DB user:
# mysqldump -u <db_user> -p paddymitra_dev_db | gzip > ~/paddymitra_dev_db_${STAMP}.sql.gz
if [ -d chroma_db ]; then
  tar czf ~/chroma_db_${STAMP}.tgz chroma_db/
  echo "   backed up chroma_db -> ~/chroma_db_${STAMP}.tgz"
else
  echo "   (no existing chroma_db to back up)"
fi

echo "== 4. Create the documents + FAQs =="
"$VENV_PY" manage.py migrate_session_content          # expect "3 created ... 192 created"
"$VENV_PY" manage.py sync_faq_vector --replace-sheets

echo "== 5. Build the fresh Chroma vector DB =="
# build_vector_db deletes and recreates the 'rice-knowledge-base' collection, so
# this IS the "new / separate" vector DB for the new code. Expect ~1,900 chunks.
"$VENV_PY" manage.py build_vector_db rice-knowledge-base

echo "== 6. Restart the app (uncomment the line that matches this server) =="
# sudo systemctl restart gunicorn
# sudo supervisorctl restart paddymitra
# touch tmp/restart.txt

echo "== 7. Smoke test (retrieval + generation) =="
"$VENV_PY" manage.py shell -c "
from catalogues.models import Catalogue
from queries.services.pipeline import retrieve, generate_answer
c=Catalogue.objects.get(slug='rice-knowledge-base')
for q in ['name some 120 days varities','How much urea per acre?','What fertiliser is used for Pooja Dhan?']:
    print('Q:',q); print('A:',generate_answer(q,retrieve(c,q,n_results=8))[:200]); print()
"
# NOTE: the FAQ-layer fixes (e.g. 'How much tillering stage fertilizer...' giving
# a DOSE, not a timing sentence) only run through the full web/WhatsApp path
# (run_pipeline_for_text), NOT through this generate_answer smoke test. Verify
# those two questions on the live app:
#   - "How much tillering stage fertilizer should I give to the rice?"  -> a dose
#   - "name some 120 days varities"                                     -> a list

echo "Done."

# ---------------------------------------------------------------------------
# OPTIONAL: zero-downtime (true side-by-side) build, if you cannot tolerate the
# few minutes of degraded retrieval while step 5 runs. It needs a one-line change
# to make CHROMA_PERSIST_DIR env-selectable in irri_ai/settings/base.py:
#     CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
# Then: build into a new dir while the old one keeps serving, verify, and swap:
#     CHROMA_PERSIST_DIR="$PROD_DIR/chroma_db_new" "$VENV_PY" manage.py build_vector_db rice-knowledge-base
#     # stop app; mv chroma_db chroma_db_old && mv chroma_db_new chroma_db; start app
# Ask and I'll wire this in and push it.
# ---------------------------------------------------------------------------
