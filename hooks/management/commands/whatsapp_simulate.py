"""Push a message through the WhatsApp path without a gateway.

    python manage.py whatsapp_simulate "DHAN how much urea per acre?"
    python manage.py whatsapp_simulate "..." --from +919876543210 --keyword DHAN
    python manage.py whatsapp_simulate --audio-url https://…/voice.ogg
    python manage.py whatsapp_simulate "..." --no-send      # do not message anyone

WHY IT EXISTS
-------------
The gateway's payload, the shared secret and a live WhatsApp number are three
things you do not have while developing, and testing through them means a real
message to a real phone. This runs the same normaliser, the same hook
resolution, the same dedupe and the same pipeline — synchronously, printing each
decision — so a broken integration can be told apart from a broken answer
before anything is deployed.

--no-send is the default-safe option for a machine with a live PICKY_ASSIST_TOKEN:
the answer is produced and printed, and nothing is delivered.
"""
from __future__ import annotations

import json
import uuid

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.test import RequestFactory

from hooks import inbound
from hooks.models import Hook, InboundMessage, WhatsAppApiKey


class Command(BaseCommand):
    help = "Run one WhatsApp message through the inbound path, synchronously."

    def add_arguments(self, parser):
        parser.add_argument("text", nargs="?", default="",
                            help="the message the farmer sent")
        parser.add_argument("--from", dest="from_number", default="+910000000000")
        parser.add_argument("--to", dest="to_number", default="",
                            help="the business number the message was sent to")
        parser.add_argument("--keyword", default="",
                            help="trigger keyword, if the gateway extracts it")
        parser.add_argument("--audio-url", default="",
                            help="treat this as a voice note at this URL")
        parser.add_argument("--message-id", default="",
                            help="reuse an id to test duplicate handling")
        parser.add_argument("--api-key", default="",
                            help="authenticate with this API key (also selects the hook)")
        parser.add_argument("--no-send", action="store_true",
                            help="run the pipeline but deliver nothing")

    def handle(self, *args, **options):
        if not (options["text"] or options["audio_url"]):
            raise CommandError("give a message, or --audio-url")
        if not Hook.objects.filter(status=Hook.STATUS_ACTIVE).exists():
            raise CommandError(
                "no ACTIVE hook exists, so every message would be ignored. "
                "Create one without leaving the terminal:\n"
                '  python manage.py whatsapp_hook --create "Paddy Mitra" --keyword DHAN\n'
                "then:\n"
                "  python manage.py whatsapp_key --create --hook DHAN")

        # A credential is needed because the endpoint fails closed, and the
        # simulator must go through the SAME door as the gateway rather than a
        # back one. When neither a key nor the shared secret is configured — a
        # fresh development box — one is minted for this run and deleted again
        # below, so the key path is genuinely exercised and nothing is left
        # behind that could be used later.
        credential = options["api_key"] or settings.PICKY_ASSIST_WEBHOOK_SECRET
        self.temporary_key = None
        payload = {
            "message_id": options["message_id"] or f"sim-{uuid.uuid4().hex[:12]}",
            "from": options["from_number"],
            "to": options["to_number"],
            "text": options["text"],
            "audio_url": options["audio_url"],
            "message_type": "audio" if options["audio_url"] else "text",
            "trigger_keyword": options["keyword"],
            # The credential goes in the body too, so this works whichever carrier
            # the deployment is configured to accept.
            "secret": credential,
        }

        message = inbound.normalise(payload)
        if options["api_key"]:
            # Exactly what the webhook will do: the key names the hook.
            key = WhatsAppApiKey.authenticate(options["api_key"])
            if key is None:
                raise CommandError("that API key is not valid, or has been revoked "
                                   "or expired — see: python manage.py whatsapp_key --list")
            hook = key.hook
            question = inbound._strip_keyword(message["text"], hook.trigger_keyword)
            self.stdout.write(f"AUTHENTICATED by key {key.masked} -> hook "
                              f"{hook.trigger_keyword} ({hook.status})")
        else:
            hook, question = inbound.resolve_hook(message)
        self.stdout.write("NORMALISED")
        for key, value in message.items():
            self.stdout.write(f"   {key:14}: {value!r}")
        self.stdout.write("ROUTED")
        self.stdout.write(f"   hook          : {hook or 'NONE — message would be ignored'}")
        self.stdout.write(f"   question       : {question!r}")
        if hook:
            self.stdout.write(f"   catalogue     : {hook.catalogue} "
                              f"(language {hook.primary_language})")
        if hook is None:
            return

        if not credential:
            self.temporary_key, credential = WhatsAppApiKey.issue(
                hook, name="whatsapp_simulate (temporary)")
            payload["secret"] = credential
            self.stdout.write(self.style.WARNING(
                f"   (no credential configured; issued a temporary key "
                f"{self.temporary_key.masked} for this run and deleting it after)"))

        request = RequestFactory().post(
            "/api/v1/whatsapp/pickyassist/", data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_API_KEY=credential or "",
        )

        if options["no_send"]:
            # Deliver nothing: the point is the answer, not the phone.
            # Patched on the whatsapp module itself, because the pipeline imports
            # the sender inside the function it calls it from — so it resolves the
            # name against this module on every send.
            from queries.services import whatsapp as sender

            def refuse_to_send(*args, **kwargs):
                self.stdout.write(self.style.WARNING(
                    "   (--no-send: gateway call skipped)"))
                return {"ok": True, "skipped": "--no-send"}

            sender.send_via_picky_assist = refuse_to_send
            sender.send_answer = refuse_to_send

        try:
            body, status = inbound.receive(request, run_inline=True)
        finally:
            # Whatever happened, the temporary credential does not outlive the
            # command that made it.
            if self.temporary_key is not None:
                self.temporary_key.delete()

        self.stdout.write("WEBHOOK RESPONSE")
        self.stdout.write(f"   http {status}: {json.dumps(body)}")

        record = InboundMessage.objects.filter(pk=body.get("inbound_id")).first()
        if record and record.query:
            query = record.query
            self.stdout.write("ANSWER")
            self.stdout.write(f"   status        : {query.status} "
                              f"in {query.response_time_seconds}s")
            self.stdout.write(f"   question (en) : {query.translated_query_text}")
            self.stdout.write(f"   answer (en)   : {query.ai_response_text}")
            self._write_local(query.ai_response_text_local)
        elif record:
            self.stdout.write(self.style.ERROR(f"   {record.status}: {record.error}"))

    def _write_local(self, text: str) -> None:
        """Print the farmer-language answer without dying on the console encoding.

        A Windows terminal is cp1252, and Odia has no representation in it — so
        printing the answer this system exists to produce raised
        UnicodeEncodeError and took the whole command down AFTER the work was
        done and the message was already answered. The answer is not worth less
        than the terminal's codepage: unprintable characters are escaped and the
        row is still shown.
        """
        label = "   answer (local):"
        try:
            self.stdout.write(f"{label} {text}")
        except UnicodeEncodeError:
            encoding = getattr(self.stdout, "encoding", None) or "ascii"
            safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
            self.stdout.write(f"{label} {safe}")
            self.stdout.write(self.style.WARNING(
                f"   (this console is {encoding}; the answer above is escaped. "
                f"Run `chcp 65001` first, or read it in the admin.)"))
