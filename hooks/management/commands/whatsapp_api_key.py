"""Manage the API keys for the GENERIC WhatsApp API (/api/v1/whatsapp/message/).

    python manage.py whatsapp_api_key --create --name "Paddy Mitra WhatsApp"
    python manage.py whatsapp_api_key --list
    python manage.py whatsapp_api_key --rotate pmw_a1b2c3d4e5f6 --grace-hours 24
    python manage.py whatsapp_api_key --revoke pmw_a1b2c3d4e5f6

These keys are NOT bound to a hook, which is what makes them work with the
provider-independent endpoint: the key authenticates the API itself, and the
message carries everything else.

`whatsapp_key` is the sibling command for HOOK-bound keys, used by the Picky
Assist webhook. The two sets are deliberately separate — revoking the credential
of one integration must not be able to shut off the other — and neither command
will touch the other's keys.

THE KEY IS SHOWN ONCE. Only a SHA-256 of it is stored, so no command can print an
existing key; if one is lost, rotate.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from hooks.models import WhatsAppApiKey


class Command(BaseCommand):
    help = "Create, list, rotate or revoke generic WhatsApp API keys."

    def add_arguments(self, parser):
        parser.add_argument("--create", action="store_true")
        parser.add_argument("--list", action="store_true")
        parser.add_argument("--rotate", metavar="PREFIX", default=None)
        parser.add_argument("--revoke", metavar="PREFIX", default=None)
        parser.add_argument("--name", default="",
                            help="which integration this key is for")
        parser.add_argument("--grace-hours", type=int,
                            default=WhatsAppApiKey.DEFAULT_GRACE_HOURS)
        parser.add_argument("--show-revoked", action="store_true")

    def handle(self, *args, **options):
        if options["create"]:
            key, plaintext = WhatsAppApiKey.issue(hook=None, name=options["name"])
            return self._announce(key, plaintext, "API key created:")
        if options["rotate"]:
            old = self._generic_key(options["rotate"])
            new, plaintext = old.rotate(grace_hours=options["grace_hours"])
            self._announce(new, plaintext, "API key rotated:")
            self.stdout.write("")
            self.stdout.write(f"  The previous key {old.masked} keeps working until "
                              f"{old.expires_at:%Y-%m-%d %H:%M} "
                              f"({options['grace_hours']}h), so the caller can be "
                              f"updated without dropping a message.")
            self.stdout.write(f"  Close that window early when it is:")
            self.stdout.write(f"     python manage.py whatsapp_api_key --revoke "
                              f"{old.prefix}")
            return
        if options["revoke"]:
            key = self._generic_key(options["revoke"])
            key.revoke()
            self.stdout.write(self.style.SUCCESS(
                f"Revoked {key.masked}. Any caller still using it now gets 401."))
            return
        return self._list(options)

    # ── helpers ───────────────────────────────────────────────────────────
    def _generic_key(self, reference: str):
        prefix = (reference or "").strip()
        if prefix.startswith(f"{WhatsAppApiKey.LABEL}_"):
            parts = prefix.split("_", 2)
            prefix = parts[1] if len(parts) > 1 else prefix
        key = WhatsAppApiKey.objects.filter(prefix=prefix).first()
        if key is None:
            raise CommandError(f"no key with prefix '{prefix}'")
        if not key.is_generic:
            raise CommandError(
                f"{key.masked} belongs to hook '{key.hook.trigger_keyword}', not to "
                f"the generic API. Use: python manage.py whatsapp_key")
        return key

    def _announce(self, key, plaintext, heading):
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(heading))
        self.stdout.write("")
        self.stdout.write(f"    {plaintext}")
        self.stdout.write("")
        self.stdout.write(self.style.WARNING(
            "  IMPORTANT: this key will not be shown again. Only a hash is stored."))
        self.stdout.write("")
        self.stdout.write("  Use it against the generic endpoint:")
        self.stdout.write("")
        self.stdout.write("    curl -X POST http://localhost:8000/api/v1/whatsapp/message/ \\")
        self.stdout.write(f"      -H 'X-API-Key: {plaintext}' \\")
        self.stdout.write("      -H 'Content-Type: application/json' \\")
        self.stdout.write("      -d '{\"phone\": \"+919876543210\", "
                          "\"message\": \"How much urea per acre?\"}'")
        self.stdout.write("")
        self.stdout.write("  Or test the whole path without WhatsApp at all:")
        self.stdout.write('    python manage.py whatsapp_api_simulate '
                          '"How much urea per acre?" --no-send')

    def _list(self, options):
        keys = WhatsAppApiKey.objects.filter(hook__isnull=True)
        if not options["show_revoked"]:
            keys = keys.filter(revoked_at__isnull=True)
        keys = list(keys)
        if not keys:
            self.stdout.write("No generic API keys yet. Create one with:")
            self.stdout.write('  python manage.py whatsapp_api_key --create '
                              '--name "Paddy Mitra WhatsApp"')
            return
        self.stdout.write(f"{'KEY':<26}{'STATE':<10}{'USED':<8}{'LAST USED':<18}NAME")
        for key in keys:
            last = f"{key.last_used_at:%Y-%m-%d %H:%M}" if key.last_used_at else "never"
            self.stdout.write(f"{key.masked:<26}{key.state:<10}"
                              f"{key.request_count:<8}{last:<18}{key.name}")
        for key in [k for k in keys if k.state == "expiring"]:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(
                f"  {key.masked} was rotated and stops working "
                f"{key.expires_at:%Y-%m-%d %H:%M}"))
