"""Create and manage WhatsApp hooks from the terminal.

    python manage.py whatsapp_hook --list
    python manage.py whatsapp_hook --create "Paddy Mitra" --keyword DHAN
    python manage.py whatsapp_hook --create "Paddy Mitra" --keyword DHAN \
        --number +919876543210 --catalogue 4 --language Odia
    python manage.py whatsapp_hook --activate DHAN
    python manage.py whatsapp_hook --pause DHAN

WHY IT EXISTS
-------------
A hook decides which catalogue answers and in which language, and until one
exists ACTIVE every inbound message is ignored — so the first thing anyone
setting this up hits is `whatsapp_simulate` refusing to run. The web form at
/hooks/new/ needs a logged-in administrator and a browser, neither of which is
available while configuring a server over SSH.

`--catalogue` may be omitted when there is exactly one catalogue: there is
nothing to choose between.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from hooks.models import Hook


class Command(BaseCommand):
    help = "Create, list, activate or pause WhatsApp hooks."

    def add_arguments(self, parser):
        parser.add_argument("--list", action="store_true")
        parser.add_argument("--create", metavar="NAME", default=None)
        parser.add_argument("--keyword", default="",
                            help="trigger word, e.g. DHAN (uppercased)")
        parser.add_argument("--number", default="",
                            help="the business WhatsApp number")
        parser.add_argument("--catalogue", type=int, default=None)
        parser.add_argument("--language", default="Odia",
                            choices=[c[0] for c in Hook.LANGUAGE_CHOICES])
        parser.add_argument("--welcome", default="",
                            help="reply sent when the farmer texts only the keyword")
        parser.add_argument("--activate", metavar="KEYWORD", default=None)
        parser.add_argument("--pause", metavar="KEYWORD", default=None)

    def handle(self, *args, **options):
        if options["create"]:
            return self._create(options)
        if options["activate"]:
            return self._set_status(options["activate"], Hook.STATUS_ACTIVE)
        if options["pause"]:
            return self._set_status(options["pause"], Hook.STATUS_PAUSED)
        return self._list()

    def _list(self):
        hooks = Hook.objects.all()
        if not hooks:
            self.stdout.write("no hooks yet — create one with:")
            self.stdout.write('  python manage.py whatsapp_hook --create "Paddy Mitra" '
                              '--keyword DHAN')
            return
        self.stdout.write(f"{'KEYWORD':<12}{'STATUS':<10}{'LANGUAGE':<10}"
                          f"{'CATALOGUE':<28}{'KEYS':<6}NAME")
        for hook in hooks:
            usable = sum(1 for k in hook.api_keys.all() if k.is_usable)
            self.stdout.write(
                f"{hook.trigger_keyword:<12}{hook.status:<10}{hook.primary_language:<10}"
                f"{str(hook.catalogue)[:26]:<28}{usable:<6}{hook.name}")

    def _create(self, options):
        from catalogues.models import Catalogue

        keyword = (options["keyword"] or "").upper().strip()
        if not keyword:
            raise CommandError("--keyword is required (e.g. --keyword DHAN)")

        catalogue_id = options["catalogue"]
        if catalogue_id:
            catalogue = Catalogue.objects.filter(id=catalogue_id).first()
            if catalogue is None:
                raise CommandError(f"no catalogue with id {catalogue_id}")
        else:
            catalogues = list(Catalogue.objects.all()[:5])
            if len(catalogues) == 1:
                catalogue = catalogues[0]
                self.stdout.write(f"using the only catalogue: {catalogue} "
                                  f"(id {catalogue.id})")
            elif not catalogues:
                raise CommandError("there are no catalogues — create one at "
                                   "/catalogues/ before a hook can answer anything")
            else:
                listing = ", ".join(f"{c.id}={c.name}" for c in catalogues)
                raise CommandError(f"several catalogues exist; pass --catalogue "
                                   f"({listing})")

        existing = Hook.objects.filter(trigger_keyword=keyword).first()
        if existing:
            raise CommandError(
                f"hook '{keyword}' already exists ({existing.name}). "
                f"Use --activate {keyword}, or pick another keyword.")

        hook = Hook.objects.create(
            name=options["create"],
            whatsapp_number=options["number"],
            trigger_keyword=keyword,
            catalogue=catalogue,
            primary_language=options["language"],
            welcome_message=options["welcome"],
            status=Hook.STATUS_ACTIVE,
        )
        self.stdout.write(self.style.SUCCESS(
            f"created hook '{hook.name}' [{hook.trigger_keyword}] — {hook.status}"))
        self.stdout.write(f"   catalogue : {hook.catalogue}")
        self.stdout.write(f"   language  : {hook.primary_language}")
        self.stdout.write(f"   endpoint  : {hook.api_endpoint}")
        self.stdout.write("")
        self.stdout.write("next: issue an API key for the gateway —")
        self.stdout.write(f"  python manage.py whatsapp_key --create --hook {keyword} "
                          f'--name "picky assist production"')
        self.stdout.write("and test it end to end without a phone —")
        self.stdout.write('  python manage.py whatsapp_simulate "how much urea per acre?" '
                          '--no-send')

    def _set_status(self, keyword, status):
        hook = Hook.objects.filter(trigger_keyword=keyword.upper().strip()).first()
        if hook is None:
            raise CommandError(f"no hook with keyword '{keyword.upper()}'")
        hook.status = status
        hook.save(update_fields=["status", "updated_at"])
        self.stdout.write(self.style.SUCCESS(
            f"hook '{hook.trigger_keyword}' is now {status}"))
