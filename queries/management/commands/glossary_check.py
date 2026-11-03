"""
glossary_check
==============
Answer one question on the machine that actually serves the site: is a glossary
term ON DISK, IN THIS PROCESS, and RECOGNISED — and if not, which of those three
failed.

Written because "I added a word and it is not found" has three completely
different causes that look identical in a browser:

    not on disk          the save never landed (permissions, a read-only deploy,
                         or the form rejected it)
    on disk, not loaded  this process is serving a stale cache
    loaded, not matched  it is there under a different spelling or normalisation

Usage:
    python manage.py glossary_check ବିଲ
    python manage.py glossary_check ବିଲ --probe     # also prove the file is writable
"""
from __future__ import annotations

import json
import os
import unicodedata

from django.core.management.base import BaseCommand

from queries.services import odia_glossary as g

PROBE_KEY = "__glossary_check_probe__"


class Command(BaseCommand):
    help = "Check whether a glossary term is on disk, loaded, and recognised."

    def add_arguments(self, parser):
        parser.add_argument("term", help="The Odia (or English) term to look for.")
        parser.add_argument(
            "--probe", action="store_true",
            help="Write a throwaway term and read it back, to prove the file is "
                 "writable by the user this process runs as. It is removed again.")

    def handle(self, *args, **opts):
        term = opts["term"]
        path = g._GLOSSARY_PATH
        ok, bad, warn = self.style.SUCCESS, self.style.ERROR, self.style.WARNING

        self.stdout.write("── file ─────────────────────────────────────────────")
        self.stdout.write(f"path      : {path}")
        self.stdout.write(f"pid       : {os.getpid()}")
        exists = path.exists()
        self.stdout.write(f"exists    : {exists}")
        if exists:
            st = path.stat()
            self.stdout.write(f"size      : {st.st_size} bytes")
            self.stdout.write(f"modified  : {self._when(st.st_mtime)}")
            self.stdout.write(
                f"writable  : {os.access(path, os.W_OK)}   "
                f"(dir: {os.access(path.parent, os.W_OK)})")
        else:
            self.stdout.write(warn("The file does not exist — the built-in "
                                   "fallback is being used and every saved term "
                                   "is going somewhere else."))

        raw = self._read(path) if exists else {}
        self.stdout.write(f"terms     : {len(raw)}")

        self.stdout.write("── the term ─────────────────────────────────────────")
        self.stdout.write(f"looking for: {term!r}")
        self.stdout.write(f"as escapes : {term.encode('unicode_escape').decode()}")

        on_disk = self._find(raw, term)
        self.stdout.write("on disk    : " + (ok(f"yes — {on_disk}") if on_disk
                                             else bad("NO")))

        variants = g.get_all_variants()
        loaded = self._find(variants, term)
        self.stdout.write("loaded here: " + (ok(f"yes — {loaded}") if loaded
                                             else bad("NO")))

        # The page's table is built from get_all_variants(), and its search box
        # matches the Odia values too — so this is exactly what the operator sees.
        self.stdout.write("in the page: " + (ok("yes") if loaded else bad("NO")))

        found = g.relevant_reverse_glossary(f"ମୋ {term}ରେ କେତେ ସାର ଦେବି?")
        self.stdout.write("recognised in a question: " +
                          (ok(f"yes — {found}") if found else bad("NO")))

        if opts["probe"]:
            self.stdout.write("── probe ────────────────────────────────────────────")
            self._probe(ok, bad)

        self.stdout.write("── verdict ──────────────────────────────────────────")
        if on_disk and loaded:
            self.stdout.write(ok("The term is saved and live. If the browser does "
                                 "not show it, you are looking at a cached page — "
                                 "reload with Ctrl+Shift+R."))
        elif on_disk and not loaded:
            self.stdout.write(bad("Saved but NOT loaded in this process: a stale "
                                  "cache. Restart the server; if it comes back, "
                                  "the deployed code predates the mtime refresh "
                                  "in reload_glossary()."))
        elif not on_disk:
            self.stdout.write(bad("NOT on disk — the save never landed. Run again "
                                  "with --probe to see whether this process can "
                                  "write the file at all, and check that the "
                                  "server serves THIS path."))

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _when(stamp):
        from datetime import datetime
        return datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M:%S")

    def _read(self, path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:                        # noqa: BLE001
            self.stdout.write(self.style.ERROR(f"unreadable JSON: {exc}"))
            return {}

    @staticmethod
    def _find(mapping, term):
        """Locate `term` as a key or a value, comparing NFD-normalised.

        Odia has two valid encodings for the same word and NFC does not unify
        the nukta letters, so a term typed on one keyboard can be a different
        byte string from the same word typed on another — which looks identical
        on screen and never matches. Normalising both sides is how the rest of
        this module compares Odia, and it is why this check does the same.
        """
        def norm(s):
            return unicodedata.normalize("NFD", str(s))

        wanted = norm(term).lower()
        for key, value in mapping.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            if norm(key).lower() == wanted:
                return f"key {key!r} -> {values}"
            for v in values:
                if norm(v).lower() == wanted:
                    return f"value of {key!r} -> {values}"
        return ""

    def _probe(self, ok, bad):
        """Write, read back and remove a throwaway term."""
        try:
            written = g.upsert_term(PROBE_KEY, "ପରୀକ୍ଷା")
        except Exception as exc:                        # noqa: BLE001
            self.stdout.write(bad(f"write RAISED: {type(exc).__name__}: {exc}"))
            self.stdout.write(bad("This process cannot write the glossary file. "
                                  "Fix the ownership/permissions of the file and "
                                  "its directory for the user the server runs as."))
            return
        readable = bool(g.get_all_variants().get(PROBE_KEY))
        self.stdout.write(f"write returned : {written}")
        self.stdout.write("read back      : " + (ok("yes") if readable else bad("NO")))
        try:
            g.delete_term(PROBE_KEY)
            self.stdout.write("cleaned up     : yes")
        except Exception as exc:                        # noqa: BLE001
            self.stdout.write(bad(f"could not remove the probe term: {exc} — "
                                  f"delete {PROBE_KEY!r} from the JSON by hand."))
        if written and readable:
            self.stdout.write(ok("The save path works on this machine."))
