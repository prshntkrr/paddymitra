"""Issue, list, rotate and revoke WhatsApp API keys.

    python manage.py whatsapp_key --list
    python manage.py whatsapp_key --create --hook DHAN --name "picky assist prod"
    python manage.py whatsapp_key --rotate pmw_a1b2c3d4e5f6 --grace-hours 24
    python manage.py whatsapp_key --rotate --hook DHAN            # newest key for a hook
    python manage.py whatsapp_key --revoke pmw_a1b2c3d4e5f6

ROTATION
--------
`--rotate` issues a replacement immediately and gives the OLD key a deadline
(24h by default). Both work until that deadline, which is the point: rotating a
webhook credential means editing the configuration at the gateway, and any
message a farmer sends between the two edits would otherwise be refused. Once
the gateway is updated, `--revoke` the old key to close the window early.

THE KEY IS SHOWN ONCE. Only a hash is stored, so there is no command that can
print it again — if it is lost, rotate.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from hooks.models import Hook, WhatsAppApiKey


class Command(BaseCommand):
    help = "Manage the API keys that authenticate WhatsApp gateways."

    def add_arguments(self, parser):
        parser.add_argument("--list", action="store_true")
        parser.add_argument("--create", action="store_true")
        parser.add_argument("--rotate", nargs="?", const="", metavar="PREFIX",
                            default=None)
        parser.add_argument("--revoke", metavar="PREFIX", default=None)
        parser.add_argument("--hook", default="", metavar="KEYWORD")
        parser.add_argument("--name", default="")
        parser.add_argument("--grace-hours", type=int,
                            default=WhatsAppApiKey.DEFAULT_GRACE_HOURS)
        parser.add_argument("--show-revoked", action="store_true")

    # ── lookup helpers ────────────────────────────────────────────────────
    def _hook(self, keyword):
        keyword = (keyword or "").upper().strip()
        if not keyword:
            hooks = list(Hook.objects.filter(status=Hook.STATUS_ACTIVE)[:5])
            if len(hooks) == 1:
                return hooks[0]
            if not hooks:
                raise CommandError(
                    "no ACTIVE hook exists. Create one first:\n"
                    '  python manage.py whatsapp_hook --create "Paddy Mitra" '
                    "--keyword DHAN")
            raise CommandError("several active hooks; pass --hook <KEYWORD> ("
                               + ", ".join(h.trigger_keyword for h in hooks) + ")")
        hook = Hook.objects.filter(trigger_keyword=keyword).first()
        if hook is None:
            raise CommandError(f"no hook with keyword '{keyword}' — "
                               f"see: python manage.py whatsapp_hook --list")
        return hook

    def _key(self, reference):
        """A key by prefix, whole key, or the newest one for a hook."""
        reference = (reference or "").strip()
        if not reference:
            hook = self._hook(self.hook_option)
            key = hook.api_keys.filter(revoked_at__isnull=True).first()
            if key is None:
                raise CommandError(f"hook '{hook.trigger_keyword}' has no key to "
                                   f"rotate — use --create")
            return key
        # Accept the full key as pasted, or just its prefix.
        prefix = reference
        if reference.startswith(f"{WhatsAppApiKey.LABEL}_"):
            parts = reference.split("_", 2)
            prefix = parts[1] if len(parts) > 1 else reference
        key = WhatsAppApiKey.objects.filter(prefix=prefix).first()
        if key is None:
            raise CommandError(f"no key with prefix '{prefix}'")
        return key

    # ── output ────────────────────────────────────────────────────────────
    def _print_new_key(self, key, plaintext, heading):
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(heading))
        self.stdout.write("")
        self.stdout.write(f"    {plaintext}")
        self.stdout.write("")
        self.stdout.write(self.style.WARNING(
            "  Copy it now — only a hash is stored, so it cannot be shown again."))
        self.stdout.write("")
        self.stdout.write(f"  hook     : {key.hook.name} [{key.hook.trigger_keyword}]"
                          f"  ({key.hook.status})")
        self.stdout.write(f"  endpoint : {key.hook.api_endpoint}")
        self.stdout.write("  use it as a header:")
        self.stdout.write(f"     X-API-Key: {plaintext}")
        self.stdout.write("  or on the URL, for a gateway that cannot set headers:")
        self.stdout.write(f"     {key.hook.api_endpoint}?api_key={plaintext}")
        self.stdout.write("")
        self.stdout.write("  test it:")
        self.stdout.write(f'     python manage.py whatsapp_simulate "how much urea per '
                          f'acre?" --api-key {plaintext} --no-send')

    def handle(self, *args, **options):
        self.hook_option = options["hook"]
        if options["create"]:
            hook = self._hook(options["hook"])
            key, plaintext = WhatsAppApiKey.issue(hook, name=options["name"])
            return self._print_new_key(key, plaintext, "API key created.")
        if options["rotate"] is not None:
            old = self._key(options["rotate"])
            new, plaintext = old.rotate(grace_hours=options["grace_hours"])
            self._print_new_key(new, plaintext, "API key rotated.")
            self.stdout.write("")
            self.stdout.write(f"  the previous key {old.masked} keeps working until "
                              f"{old.expires_at:%Y-%m-%d %H:%M} "
                              f"({options['grace_hours']}h).")
            self.stdout.write(f"  once the gateway is updated, close the window early:")
            self.stdout.write(f"     python manage.py whatsapp_key --revoke {old.prefix}")
            return
        if options["revoke"]:
            key = self._key(options["revoke"])
            key.revoke()
            self.stdout.write(self.style.SUCCESS(
                f"revoked {key.masked} ({key.hook.trigger_keyword}). Any gateway "
                f"still using it will now get 401."))
            return
        return self._list(options)

    def _list(self, options):
        keys = WhatsAppApiKey.objects.select_related("hook").all()
        if not options["show_revoked"]:
            keys = keys.filter(revoked_at__isnull=True)
        keys = list(keys)
        if not keys:
            self.stdout.write("no API keys yet. Create one with:")
            self.stdout.write("  python manage.py whatsapp_key --create --hook DHAN "
                              '--name "picky assist prod"')
            return
        self.stdout.write(f"{'KEY':<26}{'HOOK':<12}{'STATE':<10}{'USED':<8}"
                          f"{'LAST USED':<18}NAME")
        for key in keys:
            last = f"{key.last_used_at:%Y-%m-%d %H:%M}" if key.last_used_at else "never"
            # key.scope, not key.hook.trigger_keyword: a GENERIC key has no hook
            # — that is what makes it generic — and reaching through the null
            # crashed the listing outright. One generic key issued for
            # /api/v1/whatsapp/message/ made it impossible to list ANY key,
            # including the hook keys that were working fine.
            self.stdout.write(
                f"{key.masked:<26}{key.scope:<12}{key.state:<10}"
                f"{key.request_count:<8}{last:<18}{key.name}")
        expiring = [k for k in keys if k.state == "expiring"]
        for key in expiring:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(
                f"  {key.masked} was rotated and stops working "
                f"{key.expires_at:%Y-%m-%d %H:%M}"))
