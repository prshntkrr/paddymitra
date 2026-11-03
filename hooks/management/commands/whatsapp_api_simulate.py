"""Exercise the GENERIC WhatsApp API without WhatsApp, a gateway, or a hook.

    python manage.py whatsapp_api_simulate "How much urea per acre?" --no-send
    python manage.py whatsapp_api_simulate "..." --language or --phone +919876543210
    python manage.py whatsapp_api_simulate --audio-url "https://…/voice.ogg" --no-send
    python manage.py whatsapp_api_simulate "..." --message-id repeat-me   # dedupe

WHAT IT ACTUALLY DOES
---------------------
Posts through the real view, with a real generic API key, and then runs the
answering step INLINE so the answer can be printed. That means it covers the same
code the gateway does — authentication, validation, de-duplication, recording,
and the pipeline — rather than a convenient shortcut past it.

No hook is created or needed, which is the point of the endpoint under test.

--no-send produces the answer and delivers nothing, so it is safe to run on a
machine with a live PICKY_ASSIST_TOKEN.
"""
from __future__ import annotations

import json
import uuid

from django.core.management.base import BaseCommand, CommandError
from django.test import Client

from hooks import service
from hooks.models import InboundMessage, WhatsAppApiKey

URL = "/api/v1/whatsapp/message/"


class Command(BaseCommand):
    help = "Send one message through the generic WhatsApp API, end to end."

    def add_arguments(self, parser):
        parser.add_argument("text", nargs="?", default="")
        parser.add_argument("--phone", default="+910000000000")
        parser.add_argument("--language", default="",
                            help="ISO code or name; blank uses the default")
        parser.add_argument("--audio-url", default="")
        parser.add_argument("--message-id", default="")
        parser.add_argument("--catalogue", default="",
                            help="id or slug, when several exist")
        parser.add_argument("--api-key", default="",
                            help="use this key instead of a temporary one")
        parser.add_argument("--no-send", action="store_true",
                            help="produce the answer but deliver nothing")

    def handle(self, *args, **options):
        if not (options["text"] or options["audio_url"]):
            raise CommandError("give a message, or --audio-url")

        # A credential is required by the endpoint, so the simulator gets one the
        # same way a caller would rather than bypassing authentication. A
        # temporary key is deleted again below.
        temporary = None
        api_key = options["api_key"]
        if not api_key:
            temporary, api_key = WhatsAppApiKey.issue(
                hook=None, name="whatsapp_api_simulate (temporary)")

        payload = {
            "message_id": options["message_id"] or f"sim-{uuid.uuid4().hex[:12]}",
            "phone": options["phone"],
            "message_type": "audio" if options["audio_url"] else "text",
        }
        if options["text"]:
            payload["message"] = options["text"]
        if options["audio_url"]:
            payload["audio_url"] = options["audio_url"]
        if options["language"]:
            payload["language"] = options["language"]
        if options["catalogue"]:
            payload["catalogue"] = options["catalogue"]

        self.stdout.write("REQUEST")
        self.stdout.write(f"   endpoint     : POST {URL}")
        self.stdout.write(f"   phone        : {payload['phone']}")
        self.stdout.write(f"   message      : {payload.get('message', '—')!r}")
        self.stdout.write(f"   message_type : {payload['message_type']}")
        if options["audio_url"]:
            self.stdout.write(f"   audio_url    : {options['audio_url']}")
        self.stdout.write(f"   language     : "
                          f"{service.language_name(options['language']) or 'default'}")
        self.stdout.write(f"   api key      : "
                          f"{(temporary or WhatsAppApiKey.authenticate(api_key)).masked}"
                          f"{' (temporary)' if temporary else ''}")

        if options["no_send"]:
            # Deliver nothing. Patched on the whatsapp module itself, because the
            # senders are imported inside the functions that call them.
            from queries.services import whatsapp as sender

            def refuse_to_send(*args, **kwargs):
                self.stdout.write(self.style.WARNING(
                    "   (--no-send: WhatsApp delivery skipped)"))
                return {"ok": True, "skipped": "--no-send"}

            sender.send_via_picky_assist = refuse_to_send
            sender.send_answer = refuse_to_send

        # The dispatcher is replaced so the work runs HERE and its result can be
        # printed; everything before it — auth, validation, dedupe, recording —
        # runs exactly as it does for a real caller.
        queued = []
        original_dispatch = service.dispatch
        service.dispatch = lambda inbound_id: (queued.append(inbound_id) or "inline")
        try:
            response = Client().post(URL, data=json.dumps(payload),
                                     content_type="application/json",
                                     HTTP_X_API_KEY=api_key)
            self.stdout.write("")
            self.stdout.write("API RESPONSE")
            self.stdout.write(f"   http {response.status_code}: "
                              f"{response.content.decode('utf-8', 'replace')}")
            if response.status_code != 202:
                return

            inbound = InboundMessage.objects.get(pk=queued[0])
            self.stdout.write("")
            self.stdout.write("ROUTED")
            self.stdout.write(f"   provider     : {inbound.provider}")
            self.stdout.write(f"   hook         : "
                              f"{inbound.hook or 'none (no hook required)'}")
            self.stdout.write(f"   catalogue    : {service.default_catalogue(options['catalogue'] or None)}")
            self.stdout.write(f"   inbound id   : {inbound.pk}")
            self.stdout.write(f"   queued on    : "
                              f"{'celery' if service.celery_available() else 'thread pool'}"
                              f" (run inline here)")

            status = service.answer_message(inbound.pk)
            inbound.refresh_from_db()
            self.stdout.write("")
            if inbound.query is None:
                self.stdout.write(self.style.ERROR(
                    f"FAILED ({status}): {inbound.error}"))
                return
            query = inbound.query
            self.stdout.write("ANSWER")
            self.stdout.write(f"   status       : {status} / {query.status} "
                              f"in {query.response_time_seconds}s")
            self.stdout.write(f"   question (en): {query.translated_query_text}")
            self.stdout.write(f"   answer (en)  : {query.ai_response_text}")
            self._write_local(query.ai_response_text_local)
            if inbound.error:
                self.stdout.write(self.style.WARNING(f"   note         : {inbound.error}"))
        finally:
            service.dispatch = original_dispatch
            if temporary is not None:
                temporary.delete()

    def _write_local(self, text: str) -> None:
        """Print the farmer-language answer without dying on the console encoding.

        A Windows console is cp1252 and Odia has no representation in it, so
        printing the answer this system exists to produce would otherwise raise
        UnicodeEncodeError — after the work was done and paid for.
        """
        label = "   answer (loc):"
        try:
            self.stdout.write(f"{label} {text}")
        except UnicodeEncodeError:
            encoding = getattr(self.stdout, "encoding", None) or "ascii"
            safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
            self.stdout.write(f"{label} {safe}")
            self.stdout.write(self.style.WARNING(
                f"   (this console is {encoding}; escaped above. Run `chcp 65001`.)"))
