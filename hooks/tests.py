"""
The WhatsApp inbound path: authentication, payload shape, routing, dedupe.

Every test here corresponds to a way a farmer's message was lost or answered
twice. None of them needs the gateway, a phone, or an API key — the pipeline is
patched out, because what is under test is everything AROUND it.
"""
import json
import threading
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from catalogues.models import Catalogue
from hooks import inbound, service
from hooks.models import Hook, InboundMessage, WhatsAppApiKey
from queries.models import Query
from queries.services import whatsapp as sender

URL = "/api/v1/whatsapp/pickyassist/"
SECRET = "test-webhook-secret"


@override_settings(PICKY_ASSIST_WEBHOOK_SECRET=SECRET,
                   WHATSAPP_ALLOW_UNAUTHENTICATED_WEBHOOK=False,
                   PICKY_ASSIST_TOKEN="test-token")
class WebhookTestCase(TestCase):
    """Shared fixtures: one active hook, and no real pipeline."""

    def setUp(self):
        self.catalogue = Catalogue.objects.create(name="Rice")
        self.hook = Hook.objects.create(
            name="Paddy Mitra", whatsapp_number="+911234567890",
            trigger_keyword="DHAN", catalogue=self.catalogue,
            primary_language="Odia", status=Hook.STATUS_ACTIVE,
        )

    def post(self, payload, secret=SECRET, **extra):
        headers = {"HTTP_X_WEBHOOK_SECRET": secret} if secret is not None else {}
        headers.update(extra)
        return self.client.post(URL, data=json.dumps(payload),
                                content_type="application/json", **headers)

    def payload(self, **over):
        body = {"message_id": "wamid.1", "from": "+919876543210",
                "text": "DHAN how much urea per acre?"}
        body.update(over)
        return body


class WebhookAuthTests(WebhookTestCase):

    def test_a_request_without_the_secret_is_refused(self):
        # This endpoint is public and spends money on every message it accepts.
        with patch.object(inbound, "process") as process:
            response = self.post(self.payload(), secret=None)

        self.assertEqual(response.status_code, 401)
        process.assert_not_called()
        self.assertFalse(InboundMessage.objects.exists())

    def test_a_wrong_secret_is_refused(self):
        response = self.post(self.payload(), secret="not-the-secret")
        self.assertEqual(response.status_code, 401)

    def test_the_secret_is_accepted_from_a_query_parameter(self):
        # Some gateway accounts can add a query parameter but not a header.
        with patch.object(service, "dispatch"):
            response = self.client.post(
                f"{URL}?secret={SECRET}", data=json.dumps(self.payload()),
                content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "accepted")

    def test_the_secret_is_accepted_from_a_bearer_header(self):
        with patch.object(service, "dispatch"):
            response = self.post(self.payload(), secret=None,
                                 HTTP_AUTHORIZATION=f"Bearer {SECRET}")
        self.assertEqual(response.status_code, 200)

    @override_settings(PICKY_ASSIST_WEBHOOK_SECRET="")
    def test_an_unconfigured_secret_fails_closed(self):
        # Refusing every message is a loud failure; accepting anonymous
        # messages is a silent one.
        response = self.post(self.payload(), secret=None)
        self.assertEqual(response.status_code, 401)
        self.assertIn("not configured", response.json()["error"])

    @override_settings(PICKY_ASSIST_WEBHOOK_SECRET="",
                       WHATSAPP_ALLOW_UNAUTHENTICATED_WEBHOOK=True)
    def test_the_local_escape_hatch_is_explicit(self):
        with patch.object(service, "dispatch"):
            response = self.post(self.payload(), secret=None)
        self.assertEqual(response.status_code, 200)


class ApiKeyTests(WebhookTestCase):
    """Per-integration keys: authenticate, route, rotate, revoke."""

    def issue(self, hook=None, name="test"):
        return WhatsAppApiKey.issue(hook or self.hook, name=name)

    def test_a_valid_key_authenticates_and_selects_its_hook(self):
        key, plaintext = self.issue()
        with patch.object(service, "dispatch") as dispatch:
            response = self.client.post(
                URL, data=json.dumps({"message_id": "k1", "from": "+919876543210",
                                      "text": "how much urea per acre?"}),
                content_type="application/json", HTTP_X_API_KEY=plaintext)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "accepted")
        self.assertEqual(response.json()["hook"], "DHAN")
        dispatch.assert_called_once()
        self.assertEqual(InboundMessage.objects.get().hook, key.hook)

    def test_a_key_routes_without_any_trigger_keyword(self):
        # The whole point: "no active hook matched" stops being possible when the
        # credential already names the hook.
        Hook.objects.create(name="Other", whatsapp_number="+919999999999",
                            trigger_keyword="OTHER", catalogue=self.catalogue,
                            status=Hook.STATUS_ACTIVE)
        _key, plaintext = self.issue()
        with patch.object(service, "dispatch"):
            response = self.client.post(
                URL, data=json.dumps({"message_id": "k2", "from": "+91987",
                                      "text": "how much urea per acre?"}),
                content_type="application/json", HTTP_X_API_KEY=plaintext)

        self.assertEqual(response.json()["status"], "accepted")
        self.assertEqual(response.json()["hook"], "DHAN")

    def test_the_keyword_is_still_stripped_when_the_farmer_types_it(self):
        _key, plaintext = self.issue()
        with patch.object(service, "dispatch"):
            self.client.post(
                URL, data=json.dumps({"message_id": "k3", "from": "+91987",
                                      "text": "DHAN how much urea per acre?"}),
                content_type="application/json", HTTP_X_API_KEY=plaintext)

        self.assertEqual(InboundMessage.objects.get().text,
                         "how much urea per acre?")

    def test_a_key_is_accepted_from_a_query_parameter_or_bearer_header(self):
        _key, plaintext = self.issue()
        for description, kwargs, url in (
                ("query param", {}, f"{URL}?api_key={{key}}"),
                ("bearer", {"HTTP_AUTHORIZATION": "Bearer {key}"}, URL),
                ("x-api-key", {"HTTP_X_API_KEY": "{key}"}, URL)):
            with self.subTest(carrier=description):
                headers = {k: v.format(key=plaintext) for k, v in kwargs.items()}
                with patch.object(service, "dispatch"):
                    response = self.client.post(
                        url.format(key=plaintext),
                        data=json.dumps({"message_id": f"k-{description}",
                                         "from": "+91987", "text": "how much urea?"}),
                        content_type="application/json", **headers)
                self.assertEqual(response.status_code, 200, description)

    def test_an_unknown_key_is_refused(self):
        response = self.client.post(
            URL, data=json.dumps(self.payload()), content_type="application/json",
            HTTP_X_API_KEY="pmw_deadbeefcafe_nothingrealhere")
        self.assertEqual(response.status_code, 401)
        self.assertIn("API key", response.json()["error"])
        self.assertFalse(InboundMessage.objects.exists())

    def test_a_bad_key_does_not_fall_through_to_the_shared_secret(self):
        # A mistyped key must be reported as a bad key. Falling through would
        # report "missing secret", which sends the operator to the wrong place.
        response = self.client.post(
            URL, data=json.dumps(self.payload()), content_type="application/json",
            HTTP_X_API_KEY="pmw_deadbeefcafe_wrong", HTTP_X_WEBHOOK_SECRET=SECRET)
        self.assertEqual(response.status_code, 401)
        self.assertIn("API key", response.json()["error"])

    def test_the_shared_secret_still_works_for_gateways_without_a_key(self):
        with patch.object(service, "dispatch"):
            response = self.post(self.payload())
        self.assertEqual(response.json()["status"], "accepted")

    def test_a_revoked_key_is_refused(self):
        key, plaintext = self.issue()
        key.revoke()
        response = self.client.post(
            URL, data=json.dumps(self.payload()), content_type="application/json",
            HTTP_X_API_KEY=plaintext)
        self.assertEqual(response.status_code, 401)

    def test_a_key_for_a_paused_hook_is_refused_and_says_why(self):
        _key, plaintext = self.issue()
        self.hook.status = Hook.STATUS_PAUSED
        self.hook.save()
        response = self.client.post(
            URL, data=json.dumps(self.payload()), content_type="application/json",
            HTTP_X_API_KEY=plaintext)
        self.assertEqual(response.status_code, 401)
        self.assertIn("Paused", response.json()["error"])

    def test_the_plaintext_key_is_never_stored(self):
        # A database dump must contain nothing that can call this endpoint.
        _key, plaintext = self.issue()
        secret = plaintext.split("_", 2)[2]
        row = WhatsAppApiKey.objects.get()
        for field in (row.prefix, row.key_hash, row.name, str(row)):
            self.assertNotIn(secret, field)

    def test_use_is_recorded(self):
        key, plaintext = self.issue()
        with patch.object(service, "dispatch"):
            self.client.post(URL, data=json.dumps(self.payload()),
                             content_type="application/json", HTTP_X_API_KEY=plaintext)
        key.refresh_from_db()
        self.assertEqual(key.request_count, 1)
        self.assertIsNotNone(key.last_used_at)

    # ── rotation ──────────────────────────────────────────────────────────
    def test_rotation_keeps_the_old_key_working_during_the_grace_period(self):
        # Rotating means editing the gateway's configuration. Anything the farmer
        # sends between the two edits must still be answered.
        old, old_plaintext = self.issue()
        new, new_plaintext = old.rotate(grace_hours=24)

        for label, plaintext in (("old", old_plaintext), ("new", new_plaintext)):
            with self.subTest(key=label):
                with patch.object(service, "dispatch"):
                    response = self.client.post(
                        URL, data=json.dumps({"message_id": f"rot-{label}",
                                              "from": "+91987",
                                              "text": "how much urea?"}),
                        content_type="application/json", HTTP_X_API_KEY=plaintext)
                self.assertEqual(response.status_code, 200, label)
        self.assertEqual(new.rotated_from, old)
        self.assertEqual(old.state, "expiring")
        self.assertEqual(new.state, "active")

    def test_the_old_key_stops_working_once_the_grace_period_ends(self):
        old, old_plaintext = self.issue()
        old.rotate(grace_hours=0)

        response = self.client.post(
            URL, data=json.dumps(self.payload()), content_type="application/json",
            HTTP_X_API_KEY=old_plaintext)
        self.assertEqual(response.status_code, 401)
        old.refresh_from_db()
        self.assertEqual(old.state, "expired")

    def test_rotation_can_be_closed_early_by_revoking(self):
        old, old_plaintext = self.issue()
        old.rotate(grace_hours=24)
        old.revoke()

        response = self.client.post(
            URL, data=json.dumps(self.payload()), content_type="application/json",
            HTTP_X_API_KEY=old_plaintext)
        self.assertEqual(response.status_code, 401)

    def test_two_hooks_have_independent_keys(self):
        other = Hook.objects.create(
            name="Wheat", whatsapp_number="+919999999999", trigger_keyword="GAHAM",
            catalogue=self.catalogue, status=Hook.STATUS_ACTIVE)
        _k1, first = self.issue()
        _k2, second = self.issue(hook=other)

        for plaintext, expected in ((first, "DHAN"), (second, "GAHAM")):
            with self.subTest(expected=expected):
                with patch.object(service, "dispatch"):
                    response = self.client.post(
                        URL, data=json.dumps({"message_id": f"m-{expected}",
                                              "from": "+91987",
                                              "text": "how much urea?"}),
                        content_type="application/json", HTTP_X_API_KEY=plaintext)
                self.assertEqual(response.json()["hook"], expected)

    def test_health_lists_keys_without_revealing_them(self):
        _key, plaintext = self.issue()
        response = self.client.get("/api/v1/whatsapp/health/")
        body = response.json()

        self.assertEqual(body["usable_api_keys"], 1)
        self.assertEqual(body["api_keys"][0]["hook"], "DHAN")
        self.assertNotIn(plaintext.split("_", 2)[2], json.dumps(body))


class WebhookAckTests(WebhookTestCase):

    def test_the_webhook_acks_without_waiting_for_the_answer(self):
        # The gateway retries a slow webhook, and a retry used to mean a second
        # answer. Nothing may run the pipeline inside the request.
        with patch.object(service, "dispatch") as dispatch:
            response = self.post(self.payload())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "accepted")
        self.assertEqual(body["hook"], "DHAN")
        dispatch.assert_called_once()
        # Queued by id through the shared dispatcher, which is what decides
        # between Celery and the thread pool.
        self.assertEqual(dispatch.call_args[0][0], InboundMessage.objects.get().pk)

    def test_the_message_is_recorded_before_the_work_starts(self):
        # If it is not recorded first, a restart loses it with no trace.
        with patch.object(service, "dispatch"):
            self.post(self.payload())

        record = InboundMessage.objects.get()
        self.assertEqual(record.status, InboundMessage.STATUS_RECEIVED)
        self.assertEqual(record.message_id, "wamid.1")
        self.assertEqual(record.hook, self.hook)
        self.assertEqual(record.raw_payload["from"], "+919876543210")


    def test_the_submitted_job_really_runs_on_the_worker_pool(self):
        # Every other test patches dispatch away, which would hide a pool that
        # accepts work and never runs it — the farmer would get an ack and
        # silence.
        started = threading.Event()
        seen = {}

        def fake_answer(inbound_id):
            seen["id"] = inbound_id
            started.set()

        with patch.object(service, "answer_message", fake_answer):
            response = self.post(self.payload())
            ran = started.wait(timeout=10)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(ran, "the job was submitted but never executed")
        self.assertEqual(seen["id"], InboundMessage.objects.get().pk)


class WebhookDedupeTests(WebhookTestCase):

    def test_a_retried_message_is_not_answered_twice(self):
        with patch.object(service, "dispatch") as dispatch:
            first = self.post(self.payload())
            second = self.post(self.payload())          # the gateway retrying

        self.assertEqual(first.json()["status"], "accepted")
        self.assertEqual(second.json()["status"], "duplicate")
        self.assertEqual(second.status_code, 200)       # not an error to retry
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(InboundMessage.objects.count(), 1)

    def test_a_gateway_that_sends_no_id_still_dedupes_on_content(self):
        body = {"from": "+919876543210", "text": "DHAN how much urea?"}
        with patch.object(service, "dispatch") as dispatch:
            self.post(body)
            second = self.post(body)

        self.assertEqual(second.json()["status"], "duplicate")
        self.assertEqual(dispatch.call_count, 1)

    def test_the_same_question_asked_again_later_is_a_new_message(self):
        with patch.object(service, "dispatch") as dispatch:
            self.post(self.payload(message_id="a"))
            self.post(self.payload(message_id="b"))
        self.assertEqual(dispatch.call_count, 2)


class PayloadShapeTests(WebhookTestCase):
    """Field names differ per gateway account; the old view assumed just one."""

    def test_alternative_field_names_are_read(self):
        message = inbound.normalise({
            "id": "abc", "sender": "+919876543210", "receiver": "+911234567890",
            "body": "how much urea per acre?", "district": "Bargarh",
        })
        self.assertEqual(message["message_id"], "abc")
        self.assertEqual(message["from_number"], "+919876543210")
        self.assertEqual(message["to_number"], "+911234567890")
        self.assertEqual(message["text"], "how much urea per acre?")
        self.assertEqual(message["district"], "Bargarh")

    def test_a_nested_payload_is_read(self):
        message = inbound.normalise({
            "event": "message_received",
            "data": {"message_id": "x1", "mobile": "+919876543210",
                     "message": "how much urea?"},
        })
        self.assertEqual(message["message_id"], "x1")
        self.assertEqual(message["from_number"], "+919876543210")
        self.assertEqual(message["text"], "how much urea?")

    def test_a_batch_of_messages_reads_the_first(self):
        message = inbound.normalise({
            "messages": [{"id": "m1", "from": "+919876543210", "text": "hello"}]})
        self.assertEqual(message["message_id"], "m1")
        self.assertEqual(message["text"], "hello")

    def test_a_voice_note_is_recognised_from_the_declared_type(self):
        message = inbound.normalise({
            "id": "v1", "from": "+91987", "type": "ptt",
            "media_url": "https://cdn.example/voice.ogg"})
        self.assertEqual(message["message_type"], "audio")
        self.assertEqual(message["media_url"], "https://cdn.example/voice.ogg")

    def test_a_voice_note_is_recognised_from_the_url_alone(self):
        # A gateway that declares no type but attaches audio: reading this as
        # empty text answered the farmer's voice note with nothing.
        message = inbound.normalise({
            "id": "v2", "from": "+91987",
            "attachment": "https://cdn.example/note.opus"})
        self.assertEqual(message["message_type"], "audio")

    def test_a_message_with_neither_text_nor_audio_is_ignored(self):
        response = self.post({"message_id": "d1", "from": "+91987",
                              "status": "delivered"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ignored")

    def test_a_form_encoded_post_is_accepted(self):
        with patch.object(service, "dispatch"):
            response = self.client.post(
                URL, data={"message_id": "f1", "from": "+919876543210",
                           "text": "DHAN how much urea?", "secret": SECRET})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "accepted")

    def test_a_body_that_is_not_json_is_a_400(self):
        response = self.client.post(URL, data=b"\x00not json",
                                    content_type="application/json",
                                    HTTP_X_WEBHOOK_SECRET=SECRET)
        self.assertEqual(response.status_code, 400)


class HookRoutingTests(WebhookTestCase):

    def test_the_trigger_word_is_stripped_from_the_question(self):
        # "DHAN how much urea per acre?" was being retrieved for as-is, so the
        # trigger word the farmer had to type became part of their question.
        hook, question = inbound.resolve_hook(
            inbound.normalise({"from": "+91987", "text": "DHAN how much urea per acre?"}))
        self.assertEqual(hook, self.hook)
        self.assertEqual(question, "how much urea per acre?")

    def test_the_keyword_field_is_used_when_the_gateway_sends_one(self):
        hook, question = inbound.resolve_hook(inbound.normalise(
            {"from": "+91987", "trigger_keyword": "dhan", "text": "how much urea?"}))
        self.assertEqual(hook, self.hook)
        self.assertEqual(question, "how much urea?")

    def test_the_business_number_routes_when_there_is_no_keyword(self):
        other = Hook.objects.create(
            name="Other", whatsapp_number="+919999999999", trigger_keyword="OTHER",
            catalogue=self.catalogue, status=Hook.STATUS_ACTIVE)
        hook, _ = inbound.resolve_hook(inbound.normalise(
            {"from": "+91987", "to": "919999999999", "text": "how much urea?"}))
        self.assertEqual(hook, other)

    def test_a_single_active_hook_answers_anything(self):
        hook, question = inbound.resolve_hook(inbound.normalise(
            {"from": "+91987", "text": "how much urea per acre?"}))
        self.assertEqual(hook, self.hook)
        self.assertEqual(question, "how much urea per acre?")

    def test_the_trigger_word_alone_is_a_greeting_not_a_question(self):
        # "DHAN" on its own is how a farmer opens a conversation. It used to be
        # retrieved for as though it were the question, which produced a
        # confident answer to nothing.
        # A greeting never reaches the pipeline, so it is not dispatched as work —
        # the welcome reply goes straight onto the pool.
        with patch.object(service, "_pool") as pool, \
             patch.object(service, "dispatch") as dispatch, \
             patch("queries.services.pipeline.process_incoming_message") as pipeline:
            response = self.post(self.payload(text="DHAN"))

        self.assertEqual(response.json()["status"], "welcome")
        pipeline.assert_not_called()
        dispatch.assert_not_called()
        self.assertEqual(pool.return_value.submit.call_args[0][0],
                         inbound.send_welcome)

    def test_the_welcome_message_sent_is_the_hooks_own(self):
        self.hook.welcome_message = "Namaskar! Ask me about paddy."
        self.hook.save()
        record = InboundMessage.objects.create(
            provider=Hook.GATEWAY_PICKY, message_id="w1", hook=self.hook,
            from_number="+919876543210", text="")

        with patch("queries.services.whatsapp.send_via_picky_assist",
                   return_value={"ok": True}) as send:
            inbound.send_welcome(record.pk)

        self.assertEqual(send.call_args[0][2], "Namaskar! Ask me about paddy.")
        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_ANSWERED)

    def test_a_hook_with_no_welcome_message_still_replies(self):
        record = InboundMessage.objects.create(
            provider=Hook.GATEWAY_PICKY, message_id="w2", hook=self.hook,
            from_number="+919876543210", text="")

        with patch("queries.services.whatsapp.send_via_picky_assist",
                   return_value={"ok": True}) as send:
            inbound.send_welcome(record.pk)

        self.assertIn("question", send.call_args[0][2].lower())

    def test_a_paused_hook_does_not_answer(self):
        self.hook.status = Hook.STATUS_PAUSED
        self.hook.save()
        response = self.post(self.payload())
        self.assertEqual(response.json()["status"], "ignored")
        self.assertEqual(InboundMessage.objects.get().status,
                         InboundMessage.STATUS_IGNORED)

    def test_an_unmatched_keyword_with_several_hooks_is_ignored(self):
        Hook.objects.create(name="Other", whatsapp_number="+919999999999",
                            trigger_keyword="OTHER", catalogue=self.catalogue,
                            status=Hook.STATUS_ACTIVE)
        response = self.post(self.payload(text="NOPE what should I apply?"))
        self.assertEqual(response.json()["status"], "ignored")

    def test_the_advertised_endpoint_is_the_one_that_is_routed(self):
        self.assertEqual(self.hook.api_endpoint, URL)
        with patch.object(service, "dispatch"):
            self.assertEqual(self.post(self.payload()).status_code, 200)


class ProcessingTests(WebhookTestCase):

    def _record(self):
        return InboundMessage.objects.create(
            provider=Hook.GATEWAY_PICKY, message_id="p1", hook=self.hook,
            from_number="+919876543210", text="how much urea per acre?")

    def test_a_successful_run_is_marked_answered_and_linked(self):
        record = self._record()
        query = Query.objects.create(catalogue=self.catalogue, status="Responded",
                                     ai_response_text_local="Apply 25 kg/ha urea.",
                                     response_time_seconds=4.0)

        with patch("queries.services.pipeline.process_incoming_message",
                   return_value=query) as run, \
             patch("queries.services.whatsapp.send_answer",
                   return_value={"ok": True}) as send:
            inbound.process(record.pk)

        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_ANSWERED)
        self.assertEqual(record.query, query)
        self.assertEqual(record.attempts, 1)
        run.assert_called_once()
        # The text is sent by the pipeline as soon as it exists; this call adds
        # only the voice note, so nothing is delivered twice.
        self.assertTrue(send.call_args.kwargs["text_already_sent"])

    def test_a_failing_run_is_recorded_and_never_raises(self):
        # An exception on a pool thread surfaces nowhere. It has to be caught
        # and written down, or the message looks answered.
        record = self._record()
        with patch("queries.services.pipeline.process_incoming_message",
                   side_effect=RuntimeError("openai down")):
            inbound.process(record.pk)

        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_FAILED)
        self.assertIn("openai down", record.error)

    def test_the_hook_counters_are_updated(self):
        # They are shown on the hooks dashboard and were always zero.
        record = self._record()
        query = Query.objects.create(catalogue=self.catalogue, hook=self.hook,
                                     status="Responded", response_time_seconds=6.0)
        with patch("queries.services.pipeline.process_incoming_message",
                   return_value=query), \
             patch("queries.services.whatsapp.send_answer", return_value={"ok": True}):
            inbound.process(record.pk)

        self.hook.refresh_from_db()
        self.assertEqual(self.hook.messages_processed_count, 1)
        self.assertEqual(self.hook.success_rate, 100.0)
        self.assertEqual(self.hook.avg_response_time, 6.0)


class OutboundTests(TestCase):

    def setUp(self):
        self.catalogue = Catalogue.objects.create(name="Rice")
        self.hook = Hook.objects.create(
            name="Paddy Mitra", whatsapp_number="+911234567890",
            trigger_keyword="DHAN", catalogue=self.catalogue)

    def test_a_long_answer_is_split_instead_of_truncated(self):
        # WhatsApp refuses a body over 4096 characters. Slicing it lost the end
        # of the answer — often the dose.
        text = " ".join(f"Sentence number {i} about fertiliser." for i in range(200))
        pieces = sender._chunks(text, 500)

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p) <= 500 for p in pieces))
        self.assertEqual("".join(pieces.copy()).replace(" ", ""),
                         text.replace(" ", ""))

    def test_a_single_long_word_is_still_split(self):
        pieces = sender._chunks("x" * 900, 300)
        self.assertTrue(all(len(p) <= 300 for p in pieces))
        self.assertEqual(sum(len(p) for p in pieces), 900)

    @override_settings(PICKY_ASSIST_TOKEN="t", PICKY_ASSIST_APPLICATION="121",
                       PICKY_ASSIST_PRIORITY="1", WHATSAPP_MAX_CHARS=500)
    def test_the_body_is_the_shape_the_gateway_accepts(self):
        with patch.object(sender, "_post", return_value={"ok": True}) as post:
            result = sender.send_via_picky_assist(
                self.hook, "+919876543210", "Apply 25 kg/ha urea.",
                audio_url="https://cdn.example/a.mp3")

        self.assertTrue(result["ok"])
        # One call: the text and its voice note travel together.
        self.assertEqual(len(post.call_args_list), 1)
        self.assertEqual(post.call_args_list[0].args[0], {
            "token": "t",
            "priority": "1",
            "application": "121",
            "globalmedia": "https://cdn.example/a.mp3",
            "data": [{"number": "+919876543210",
                      "message": "Apply 25 kg/ha urea."}],
        })

    @override_settings(PICKY_ASSIST_TOKEN="t", PICKY_ASSIST_APPLICATION="121")
    def test_globalmedia_is_present_but_empty_when_there_is_no_audio(self):
        # The receiving script reads the key unconditionally, so it is always sent.
        with patch.object(sender, "_post", return_value={"ok": True}) as post:
            sender.send_via_picky_assist(self.hook, "+91987", "Apply urea.")
        self.assertEqual(post.call_args_list[0].args[0]["globalmedia"], "")

    @override_settings(PICKY_ASSIST_TOKEN="t", PICKY_ASSIST_APPLICATION="121",
                       WHATSAPP_MAX_CHARS=500)
    def test_a_split_answer_does_not_repeat_the_voice_note(self):
        # globalmedia applies to every entry in a batch, so attaching it to each
        # part would send the same recording three times.
        long_answer = " ".join(f"Sentence {i} about fertiliser." for i in range(120))
        with patch.object(sender, "_post", return_value={"ok": True}) as post:
            sender.send_via_picky_assist(self.hook, "+91987", long_answer,
                                         audio_url="https://cdn.example/a.mp3")

        bodies = [call.args[0] for call in post.call_args_list]
        with_media = [b for b in bodies if b["globalmedia"]]
        self.assertEqual(len(with_media), 1)
        self.assertEqual(with_media[0]["data"][0]["message"], "")
        self.assertGreater(len(bodies), 2)

    @override_settings(PICKY_ASSIST_TOKEN="t", PICKY_ASSIST_APPLICATION="121")
    def test_an_audio_only_send_still_produces_one_call(self):
        # The hook path: the pipeline already sent the text, so this call carries
        # only the voice note. It used to send nothing at all and report failure.
        with patch.object(sender, "_post", return_value={"ok": True}) as post:
            result = sender.send_via_picky_assist(
                self.hook, "+91987", "", audio_url="https://cdn.example/a.mp3")

        self.assertTrue(result["ok"])
        self.assertEqual(len(post.call_args_list), 1)
        body = post.call_args_list[0].args[0]
        self.assertEqual(body["globalmedia"], "https://cdn.example/a.mp3")
        self.assertEqual(body["data"][0]["message"], "")

    @override_settings(PICKY_ASSIST_TOKEN="t", PICKY_ASSIST_APPLICATION="")
    def test_a_missing_application_id_fails_loudly(self):
        # Without it the gateway returns 200 and delivers nothing, which is the
        # one outcome that must not look like success.
        with self.assertRaises(RuntimeError) as caught:
            sender.send_via_picky_assist(self.hook, "+91987", "Apply urea.")
        self.assertIn("PICKY_ASSIST_APPLICATION", str(caught.exception))

    @override_settings(PICKY_ASSIST_TOKEN="t", PICKY_ASSIST_APPLICATION="121")
    def test_a_transient_gateway_error_is_retried(self):
        responses = [
            type("R", (), {"status_code": 503, "json": lambda self: {"e": 1},
                           "text": ""})(),
            type("R", (), {"status_code": 200, "json": lambda self: {"ok": 1},
                           "text": ""})(),
        ]
        with patch("queries.services.whatsapp.requests.post",
                   side_effect=responses) as post, \
             patch("queries.services.whatsapp.time.sleep"):
            result = sender.send_via_picky_assist(self.hook, "+91987", "hi")

        self.assertTrue(result["ok"])
        self.assertEqual(post.call_count, 2)

    @override_settings(PICKY_ASSIST_TOKEN="t", PICKY_ASSIST_APPLICATION="121")
    def test_a_permanent_gateway_error_is_not_retried(self):
        bad = type("R", (), {"status_code": 401, "json": lambda self: {"e": "auth"},
                             "text": ""})()
        with patch("queries.services.whatsapp.requests.post",
                   return_value=bad) as post:
            result = sender.send_via_picky_assist(self.hook, "+91987", "hi")

        self.assertFalse(result["ok"])
        self.assertEqual(post.call_count, 1)

    @override_settings(PUBLIC_BASE_URL="", WHATSAPP_SEND_VOICE=True)
    def test_no_public_base_url_means_no_broken_audio_link(self):
        query = Query.objects.create(catalogue=self.catalogue)
        query.ai_response_voice_file.name = "audio/out/q1.mp3"
        self.assertEqual(sender.voice_url_for(query), "")

    @override_settings(PUBLIC_BASE_URL="https://paddymitra.example.org",
                       WHATSAPP_SEND_VOICE=True, MEDIA_URL="/media/")
    def test_the_voice_url_is_absolute_when_the_origin_is_known(self):
        query = Query.objects.create(catalogue=self.catalogue)
        query.ai_response_voice_file.name = "audio/out/q1.mp3"
        self.assertEqual(sender.voice_url_for(query),
                         "https://paddymitra.example.org/media/audio/out/q1.mp3")


class HealthTests(WebhookTestCase):

    def test_health_reports_configuration_without_leaking_it(self):
        response = self.client.get("/api/v1/whatsapp/health/")
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["ok"])
        self.assertTrue(body["webhook_secret_configured"])
        self.assertEqual(body["inbound_endpoint"], URL)
        self.assertEqual(body["active_hooks"][0]["keyword"], "DHAN")
        self.assertNotIn(SECRET, json.dumps(body))
        self.assertNotIn("test-token", json.dumps(body))
