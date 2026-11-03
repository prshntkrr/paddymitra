"""The WhatsApp API console: several keys, rotation, revocation.

The model could already do all three; what is tested here is that the screen
does them safely — the plaintext shown exactly once, rotation leaving the old
key alive for its grace period, revocation taking effect immediately, and none
of it available to someone without an editor role.
"""
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from accounts.models import Role
from catalogues.models import Catalogue
from hooks.models import Hook, WhatsAppApiKey


class ConsoleTestCase(TestCase):

    def setUp(self):
        self.catalogue = Catalogue.objects.create(name="Rice", status="Published")
        self.hook = Hook.objects.create(
            name="Paddy Mitra", whatsapp_number="+919999999999",
            trigger_keyword="DHAN", catalogue=self.catalogue,
            status=Hook.STATUS_ACTIVE)
        editor, _ = Role.objects.get_or_create(name="Editor")
        viewer, _ = Role.objects.get_or_create(name="Viewer")
        User = get_user_model()
        # This project logs people in by email, not username.
        self.editor = User.objects.create_user(
            "editor@example.org", password="pw", username="editor", role=editor)
        self.viewer = User.objects.create_user(
            "viewer@example.org", password="pw", username="viewer", role=viewer)

    def as_editor(self):
        self.client.force_login(self.editor)

    def create(self, **data):
        return self.client.post(reverse("api_endpoints:whatsapp_key_create"), data)


class VisibilityTests(ConsoleTestCase):

    def test_the_whatsapp_api_is_listed_on_the_api_page(self):
        self.as_editor()
        response = self.client.get(reverse("api_endpoints:list"))

        self.assertContains(response, "WhatsApp API")
        self.assertContains(response, reverse("api_endpoints:whatsapp"))

    def test_the_console_shows_the_three_endpoints(self):
        self.as_editor()
        response = self.client.get(reverse("api_endpoints:whatsapp"))

        self.assertContains(response, "/api/v1/whatsapp/pickyassist/")
        self.assertContains(response, "/api/v1/whatsapp/message/")
        self.assertContains(response, "/api/v1/whatsapp/health/")

    def test_login_is_required(self):
        response = self.client.get(reverse("api_endpoints:whatsapp"))

        self.assertEqual(302, response.status_code)


class CreateTests(ConsoleTestCase):

    def test_a_key_can_be_created_for_a_hook(self):
        self.as_editor()
        self.create(name="picky prod", hook=self.hook.pk)

        key = WhatsAppApiKey.objects.get()
        self.assertEqual(self.hook, key.hook)
        self.assertEqual("picky prod", key.name)

    def test_a_generic_key_has_no_hook(self):
        self.as_editor()
        self.create(name="partner app", hook="")

        self.assertTrue(WhatsAppApiKey.objects.get().is_generic)

    def test_several_keys_can_exist_for_one_hook(self):
        # One per integration is easier to revoke than one shared everywhere.
        self.as_editor()
        self.create(name="gateway", hook=self.hook.pk)
        self.create(name="staging", hook=self.hook.pk)

        self.assertEqual(2, WhatsAppApiKey.objects.filter(hook=self.hook).count())

    def test_the_plaintext_is_shown_once_and_not_again(self):
        self.as_editor()
        self.create(name="once", hook=self.hook.pk)

        shown = self.client.get(reverse("api_endpoints:whatsapp"))
        self.assertContains(shown, "Copy this key now")
        plaintext = WhatsAppApiKey.objects.get().prefix
        self.assertContains(shown, plaintext)

        # Reloading must not repeat it: the secret half is not stored at all.
        again = self.client.get(reverse("api_endpoints:whatsapp"))
        self.assertNotContains(again, "Copy this key now")

    def test_a_viewer_cannot_create_a_key(self):
        self.client.force_login(self.viewer)
        self.create(name="nope", hook=self.hook.pk)

        self.assertEqual(0, WhatsAppApiKey.objects.count())


class RotateTests(ConsoleTestCase):

    def rotate(self, key, **data):
        return self.client.post(
            reverse("api_endpoints:whatsapp_key_rotate", args=[key.pk]), data)

    def test_rotating_issues_a_replacement(self):
        self.as_editor()
        old, _ = WhatsAppApiKey.issue(hook=self.hook, name="gateway")
        self.rotate(old, grace_hours=24)

        self.assertEqual(2, WhatsAppApiKey.objects.count())
        new = WhatsAppApiKey.objects.exclude(pk=old.pk).get()
        self.assertEqual(old, new.rotated_from)

    def test_the_old_key_keeps_working_during_the_grace_period(self):
        # Rotating means editing the gateway's configuration; a farmer's message
        # sent between the two edits must not be refused.
        self.as_editor()
        old, plaintext = WhatsAppApiKey.issue(hook=self.hook, name="gateway")
        self.rotate(old, grace_hours=24)

        old.refresh_from_db()
        self.assertGreater(old.expires_at, timezone.now())
        self.assertEqual("expiring", old.state)
        self.assertIsNotNone(WhatsAppApiKey.authenticate(plaintext))

    def test_a_viewer_cannot_rotate(self):
        self.client.force_login(self.viewer)
        old, _ = WhatsAppApiKey.issue(hook=self.hook, name="gateway")
        self.rotate(old, grace_hours=24)

        self.assertEqual(1, WhatsAppApiKey.objects.count())


class RevokeTests(ConsoleTestCase):

    def revoke(self, key):
        return self.client.post(
            reverse("api_endpoints:whatsapp_key_revoke", args=[key.pk]))

    def test_revoking_takes_effect_at_once(self):
        self.as_editor()
        key, plaintext = WhatsAppApiKey.issue(hook=self.hook, name="leaked")
        self.revoke(key)

        key.refresh_from_db()
        self.assertEqual("revoked", key.state)
        self.assertIsNone(WhatsAppApiKey.authenticate(plaintext))

    def test_a_revoked_key_is_hidden_until_asked_for(self):
        self.as_editor()
        key, _ = WhatsAppApiKey.issue(hook=self.hook, name="leaked")
        self.revoke(key)

        # On the key list, not in the HTML: the confirmation message names the
        # key it just revoked, and that is the page telling you what it did.
        hidden = self.client.get(reverse("api_endpoints:whatsapp"))
        self.assertEqual([], list(hidden.context["keys"]))
        shown = self.client.get(reverse("api_endpoints:whatsapp") + "?revoked=1")
        self.assertEqual([key], list(shown.context["keys"]))

    def test_a_viewer_cannot_revoke(self):
        self.client.force_login(self.viewer)
        key, _ = WhatsAppApiKey.issue(hook=self.hook, name="leaked")
        self.revoke(key)

        key.refresh_from_db()
        self.assertEqual("active", key.state)
