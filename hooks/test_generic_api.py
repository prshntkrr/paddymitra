"""
The provider-independent WhatsApp API: POST /api/v1/whatsapp/message/

What these tests hold down, in order of how much it would cost to get wrong:

    no RAG call inside the request      · a slow response is retried, and each
                                          retry is another paid answer
    one RAG call per message            · a duplicate must cost nothing
    no hook anywhere                    · the whole point of the endpoint
    an answer is never lost             · a delivery failure must not discard
                                          work that has already been paid for
    a key is never stored in plaintext  · a database dump must be useless

The pipeline itself is patched out throughout: what is under test is everything
around it, and the pipeline has its own tests.
"""
import json
from unittest.mock import patch

from django.test import TestCase, override_settings

from catalogues.models import Catalogue
from hooks import service
from hooks.models import Hook, InboundMessage, WhatsAppApiKey
from queries.models import Query

URL = "/api/v1/whatsapp/message/"
HEALTH = "/api/v1/whatsapp/health/"


class GenericApiTestCase(TestCase):
    """One catalogue, one generic key, and no hook at all."""

    def setUp(self):
        self.catalogue = Catalogue.objects.create(name="Rice", status="Published")
        self.key, self.plaintext = WhatsAppApiKey.issue(hook=None, name="test")

    def post(self, payload, key="__default__", **extra):
        headers = {} if key is None else {
            "HTTP_X_API_KEY": self.plaintext if key == "__default__" else key}
        headers.update(extra)
        return self.client.post(URL, data=json.dumps(payload),
                                content_type="application/json", **headers)

    def body(self, **over):
        payload = {"message_id": "msg_123", "phone": "+919876543210",
                   "message": "How much urea should I apply per acre?"}
        payload.update(over)
        return payload

    def answered_query(self, **over):
        fields = {"catalogue": self.catalogue, "status": "Responded",
                  "ai_response_text": "Apply 20-25 kg urea per acre.",
                  "ai_response_text_local": "…", "response_time_seconds": 5.0}
        fields.update(over)
        return Query.objects.create(**fields)


class RequestTests(GenericApiTestCase):

    def test_a_valid_text_request_is_accepted_and_queued(self):
        with patch.object(service, "dispatch") as dispatch:
            response = self.post(self.body())

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(),
                         {"status": "accepted", "message_id": "msg_123"})
        dispatch.assert_called_once_with(InboundMessage.objects.get().pk)

    def test_the_pipeline_never_runs_inside_the_request(self):
        # A request that takes 20 seconds is retried by every gateway there is,
        # and each retry is another paid answer.
        with patch.object(service, "dispatch"), \
             patch("queries.services.pipeline.run_pipeline_for_text") as rag:
            self.post(self.body())
        rag.assert_not_called()

    def test_phone_is_required(self):
        response = self.post({"message": "how much urea?"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "phone is required")
        self.assertFalse(InboundMessage.objects.exists())

    def test_message_is_required(self):
        response = self.post({"phone": "+919876543210"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "message is required")

    def test_a_malformed_request_is_not_silently_accepted(self):
        for description, payload in (("empty", {}), ("blank message",
                                                     {"phone": "+91987",
                                                      "message": "   "})):
            with self.subTest(case=description):
                self.assertEqual(self.post(payload).status_code, 400)

    def test_a_body_that_is_not_json_is_a_400(self):
        response = self.client.post(URL, data=b"\x00 not json",
                                    content_type="application/json",
                                    HTTP_X_API_KEY=self.plaintext)
        self.assertEqual(response.status_code, 400)

    def test_an_unknown_message_type_is_a_400(self):
        response = self.post(self.body(message_type="sticker"))
        self.assertEqual(response.status_code, 400)

    def test_message_type_defaults_to_text(self):
        with patch.object(service, "dispatch"):
            self.post({"phone": "+91987", "message": "how much urea?"})
        self.assertEqual(InboundMessage.objects.get().message_type, "text")

    def test_a_missing_message_id_is_generated(self):
        with patch.object(service, "dispatch"):
            response = self.post({"phone": "+91987", "message": "how much urea?"})

        self.assertEqual(response.status_code, 202)
        generated = response.json()["message_id"]
        self.assertTrue(generated)
        self.assertEqual(InboundMessage.objects.get().message_id, generated)

    def test_a_get_request_is_rejected(self):
        self.assertEqual(self.client.get(URL).status_code, 405)


class SynchronousTests(GenericApiTestCase):
    """wait=true: hold the connection and return the answer in the response."""

    def test_wait_returns_the_answer_in_the_response(self):
        query = self.answered_query(
            ai_response_text="Apply 20-25 kg urea per acre.",
            ai_response_text_local="ପ୍ରତି ଏକରରେ ୨୦-୨୫ କିଲୋ ୟୁରିଆ ପ୍ରୟୋଗ କରନ୍ତୁ।",
            translated_query_text="How much urea per acre?",
            original_query_language="Odia")

        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=query), \
             patch("queries.services.whatsapp.send_answer",
                   return_value={"ok": True}), \
             patch.object(service, "dispatch") as dispatch:
            response = self.post(self.body(wait=True))

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "answered")
        self.assertEqual(body["answer"], "ପ୍ରତି ଏକରରେ ୨୦-୨୫ କିଲୋ ୟୁରିଆ ପ୍ରୟୋଗ କରନ୍ତୁ।")
        self.assertEqual(body["answer_en"], "Apply 20-25 kg urea per acre.")
        self.assertEqual(body["question_en"], "How much urea per acre?")
        self.assertEqual(body["query_id"], query.pk)
        self.assertTrue(body["delivered"])
        # Nothing was queued: it ran here instead.
        dispatch.assert_not_called()

    def test_without_wait_the_answer_is_not_in_the_response(self):
        with patch.object(service, "dispatch"):
            response = self.post(self.body())
        self.assertEqual(response.status_code, 202)
        self.assertNotIn("answer", response.json())

    def test_deliver_false_produces_the_answer_and_sends_nothing(self):
        # For a caller that puts the message on WhatsApp itself. Without this,
        # both would send and the farmer would get the same advice twice.
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=self.answered_query()), \
             patch("queries.services.whatsapp.send_answer") as send:
            response = self.post(self.body(wait=True, deliver=False))

        self.assertEqual(response.status_code, 200)
        send.assert_not_called()
        self.assertFalse(response.json()["delivered"])
        self.assertEqual(response.json()["status"], "answered")
        # The answer still exists and is still recorded against the message.
        self.assertIsNotNone(InboundMessage.objects.get().query_id)

    def test_a_failure_while_waiting_is_a_500_with_the_reason(self):
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   side_effect=RuntimeError("openai down")):
            response = self.post(self.body(wait=True))

        self.assertEqual(response.status_code, 500)
        self.assertIn("openai down", response.json()["error"])
        self.assertEqual(response.json()["message_id"], "msg_123")
        # Recorded, so whatsapp_retry can still pick it up.
        self.assertEqual(InboundMessage.objects.get().status,
                         InboundMessage.STATUS_FAILED)

    def test_a_retry_after_a_timeout_returns_the_answer_not_a_second_one(self):
        # A caller that gives up waiting and retries must not pay twice.
        query = self.answered_query()
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=query) as rag, \
             patch("queries.services.whatsapp.send_answer", return_value={"ok": True}):
            first = self.post(self.body(wait=True))
            second = self.post(self.body(wait=True))

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["status"], "duplicate")
        self.assertEqual(second.json()["answer_en"], query.ai_response_text)
        rag.assert_called_once()

    def test_the_flag_is_accepted_as_a_string_too(self):
        # A form post, or a caller that sends "true" rather than true.
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=self.answered_query()), \
             patch("queries.services.whatsapp.send_answer", return_value={"ok": True}):
            response = self.post(self.body(wait="true"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", response.json())


class AudioRequestTests(GenericApiTestCase):

    def audio(self, **over):
        payload = {"message_id": "msg_456", "phone": "+919876543210",
                   "message_type": "audio",
                   "audio_url": "https://example.com/audio/voice.ogg",
                   "language": "or"}
        payload.update(over)
        return payload

    def test_an_audio_request_is_accepted(self):
        with patch.object(service, "dispatch"):
            response = self.post(self.audio())

        self.assertEqual(response.status_code, 202)
        record = InboundMessage.objects.get()
        self.assertEqual(record.message_type, "audio")
        self.assertEqual(record.media_url, "https://example.com/audio/voice.ogg")
        self.assertEqual(record.language, "Odia")      # ISO code translated once

    def test_audio_url_is_required_for_an_audio_request(self):
        response = self.post(self.audio(audio_url=""))
        self.assertEqual(response.status_code, 400)
        self.assertIn("audio_url is required", response.json()["error"])

    def test_audio_goes_through_the_existing_sarvam_transcription(self):
        # Not a second transcription implementation: the same entry point the
        # Picky Assist path uses, which is where the Sarvam / Whisper routing is.
        record = InboundMessage.objects.create(
            provider=InboundMessage.PROVIDER_GENERIC, message_id="a1",
            from_number="+91987", message_type="audio",
            media_url="https://example.com/voice.ogg", language="Odia")

        with patch("queries.services.pipeline.fetch_audio",
                   return_value=(b"audio", ".ogg")), \
             patch("queries.services.pipeline.transcribe_audio_bytes",
                   return_value=("kete urea", "sarvam_odia")) as stt, \
             patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=self.answered_query()) as rag, \
             patch("queries.services.whatsapp.send_answer", return_value={"ok": True}):
            service.answer_message(record.pk)

        stt.assert_called_once_with(b"audio", ".ogg", language_hint="Odia")
        self.assertEqual(rag.call_args.kwargs["query_text"], "kete urea")
        self.assertEqual(rag.call_args.kwargs["initial_trace"],
                         {"stt_backend": "sarvam_odia"})

    def test_the_farmers_voice_note_is_kept_not_just_its_transcript(self):
        # A transcript that reads oddly is the one time the recording matters,
        # and the WhatsApp path used to delete it with the temporary file.
        record = InboundMessage.objects.create(
            provider=InboundMessage.PROVIDER_GENERIC, message_id="keep-1",
            from_number="+91987", message_type="audio",
            media_url="https://example.com/voice.ogg", language="Odia")
        query = self.answered_query()

        with patch("queries.services.pipeline.fetch_audio",
                   return_value=(b"OggS-fake-voice-note", ".ogg")) as fetch, \
             patch("queries.services.pipeline.transcribe_audio_bytes",
                   return_value=("kete urea", "sarvam_odia")) as stt, \
             patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=query) as rag, \
             patch("queries.services.whatsapp.send_answer", return_value={"ok": True}):
            service.answer_message(record.pk)

        # Downloaded once, and the same bytes are transcribed and stored.
        fetch.assert_called_once_with("https://example.com/voice.ogg")
        self.assertEqual(stt.call_args[0][0], b"OggS-fake-voice-note")
        self.assertEqual(rag.call_args.kwargs["source_audio"],
                         (b"OggS-fake-voice-note", ".ogg"))

    def test_a_text_message_stores_no_audio(self):
        record = InboundMessage.objects.create(
            provider=InboundMessage.PROVIDER_GENERIC, message_id="keep-2",
            from_number="+91987", text="how much urea per acre?")
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=self.answered_query()) as rag, \
             patch("queries.services.whatsapp.send_answer", return_value={"ok": True}):
            service.answer_message(record.pk)
        self.assertIsNone(rag.call_args.kwargs["source_audio"])

    def test_an_untranscribable_voice_note_is_recorded_as_failed(self):
        record = InboundMessage.objects.create(
            provider=InboundMessage.PROVIDER_GENERIC, message_id="a2",
            from_number="+91987", message_type="audio",
            media_url="https://example.com/silence.ogg")

        with patch("queries.services.pipeline.fetch_audio",
                   return_value=(b"audio", ".ogg")), \
             patch("queries.services.pipeline.transcribe_audio_bytes",
                   return_value=("", "sarvam_odia")), \
             patch("queries.services.pipeline.run_pipeline_for_text") as rag:
            service.answer_message(record.pk)

        rag.assert_not_called()
        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_FAILED)
        self.assertIn("transcribed", record.error)


class AuthenticationTests(GenericApiTestCase):

    def test_a_missing_key_is_refused(self):
        response = self.post(self.body(), key=None)
        self.assertEqual(response.status_code, 401)
        self.assertIn("missing API key", response.json()["error"])
        self.assertFalse(InboundMessage.objects.exists())

    def test_an_invalid_key_is_refused(self):
        response = self.post(self.body(), key="pmw_deadbeefcafe_nothingreal")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"], "invalid API key")

    def test_a_revoked_key_is_refused(self):
        self.key.revoke()
        self.assertEqual(self.post(self.body()).status_code, 401)

    def test_an_expired_key_is_refused(self):
        self.key.rotate(grace_hours=0)
        self.assertEqual(self.post(self.body()).status_code, 401)

    def test_a_rotated_key_and_its_replacement_both_work_during_the_grace(self):
        _new, replacement = self.key.rotate(grace_hours=24)
        for label, plaintext in (("old", self.plaintext), ("new", replacement)):
            with self.subTest(key=label):
                with patch.object(service, "dispatch"):
                    response = self.post(self.body(message_id=f"g-{label}"),
                                         key=plaintext)
                self.assertEqual(response.status_code, 202, label)

    def test_a_hook_bound_key_cannot_be_used_here(self):
        # The two surfaces are separate on purpose: revoking one integration's
        # credential must not be able to shut off the other.
        hook = Hook.objects.create(name="H", whatsapp_number="+911",
                                   trigger_keyword="DHAN", catalogue=self.catalogue,
                                   status=Hook.STATUS_ACTIVE)
        _key, hook_plaintext = WhatsAppApiKey.issue(hook=hook, name="hook key")

        response = self.post(self.body(), key=hook_plaintext)
        self.assertEqual(response.status_code, 401)

    def test_a_key_is_accepted_from_every_documented_carrier(self):
        for label, kwargs in (
                ("x-api-key", {"HTTP_X_API_KEY": self.plaintext}),
                ("bearer", {"HTTP_AUTHORIZATION": f"Bearer {self.plaintext}"}),
                ("token", {"HTTP_AUTHORIZATION": f"Token {self.plaintext}"})):
            with self.subTest(carrier=label):
                with patch.object(service, "dispatch"):
                    response = self.client.post(
                        URL, data=json.dumps(self.body(message_id=f"c-{label}")),
                        content_type="application/json", **kwargs)
                self.assertEqual(response.status_code, 202, label)

    def test_the_plaintext_key_is_never_stored(self):
        secret = self.plaintext.split("_", 2)[2]
        row = WhatsAppApiKey.objects.get(pk=self.key.pk)
        for value in (row.prefix, row.key_hash, row.name, str(row), row.masked):
            self.assertNotIn(secret, value)
        # And it is a hash of the secret, not an encoding of it.
        self.assertEqual(len(row.key_hash), 64)

    def test_use_is_recorded_on_the_key(self):
        with patch.object(service, "dispatch"):
            self.post(self.body())
        self.key.refresh_from_db()
        self.assertEqual(self.key.request_count, 1)
        self.assertIsNotNone(self.key.last_used_at)

    def test_no_webhook_secret_is_needed(self):
        # The generic API is authenticated by its own key and must not depend on
        # the Picky Assist deployment secret.
        with override_settings(PICKY_ASSIST_WEBHOOK_SECRET=""):
            with patch.object(service, "dispatch"):
                self.assertEqual(self.post(self.body()).status_code, 202)


class DuplicateTests(GenericApiTestCase):

    def test_the_same_message_id_is_not_processed_twice(self):
        with patch.object(service, "dispatch") as dispatch:
            first = self.post(self.body())
            second = self.post(self.body())

        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["status"], "accepted")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json(),
                         {"status": "duplicate", "message_id": "msg_123"})
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(InboundMessage.objects.count(), 1)

    def test_two_different_messages_are_both_processed(self):
        with patch.object(service, "dispatch") as dispatch:
            self.post(self.body(message_id="a"))
            self.post(self.body(message_id="b"))
        self.assertEqual(dispatch.call_count, 2)

    def test_two_messages_with_the_same_text_are_not_duplicates(self):
        # Generated ids must not collide: a farmer asking the same thing twice is
        # asking twice.
        with patch.object(service, "dispatch") as dispatch:
            self.post({"phone": "+91987", "message": "how much urea?"})
            self.post({"phone": "+91987", "message": "how much urea?"})
        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual(InboundMessage.objects.count(), 2)

    def test_the_same_id_from_a_different_provider_is_not_a_duplicate(self):
        InboundMessage.objects.create(provider=Hook.GATEWAY_PICKY,
                                      message_id="msg_123", from_number="+91987")
        with patch.object(service, "dispatch") as dispatch:
            response = self.post(self.body())
        self.assertEqual(response.status_code, 202)
        self.assertEqual(dispatch.call_count, 1)

    def test_a_duplicate_never_reaches_the_pipeline(self):
        with patch.object(service, "dispatch"):
            self.post(self.body())
        with patch("queries.services.pipeline.run_pipeline_for_text") as rag:
            self.post(self.body())
        rag.assert_not_called()


class NoHookTests(GenericApiTestCase):

    def test_no_hook_is_required_and_none_is_created(self):
        self.assertFalse(Hook.objects.exists())
        with patch.object(service, "dispatch"):
            response = self.post(self.body())

        self.assertEqual(response.status_code, 202)
        self.assertFalse(Hook.objects.exists(), "a hook was invented")

    def test_the_message_is_stored_with_a_null_hook_and_generic_provider(self):
        with patch.object(service, "dispatch"):
            self.post(self.body())

        record = InboundMessage.objects.get()
        self.assertIsNone(record.hook)
        self.assertEqual(record.provider, "generic")
        self.assertEqual(record.from_number, "+919876543210")
        self.assertEqual(record.status, InboundMessage.STATUS_RECEIVED)
        self.assertEqual(record.raw_payload["message"],
                         "How much urea should I apply per acre?")

    def test_a_paused_hook_does_not_affect_the_generic_api(self):
        Hook.objects.create(name="H", whatsapp_number="+911", trigger_keyword="DHAN",
                            catalogue=self.catalogue, status=Hook.STATUS_PAUSED)
        with patch.object(service, "dispatch"):
            self.assertEqual(self.post(self.body()).status_code, 202)

    @override_settings(WHATSAPP_DEFAULT_CATALOGUE_ID=None)
    def test_a_deployment_with_no_catalogue_says_so_instead_of_dropping_it(self):
        Catalogue.objects.all().delete()
        response = self.post(self.body())
        self.assertEqual(response.status_code, 503)
        self.assertIn("catalogue", response.json()["error"])
        self.assertFalse(InboundMessage.objects.exists())

    @override_settings(WHATSAPP_DEFAULT_CATALOGUE_ID=None)
    def test_several_catalogues_require_one_to_be_chosen(self):
        Catalogue.objects.create(name="Wheat", status="Published")
        response = self.post(self.body())
        self.assertEqual(response.status_code, 503)

        # …and naming one in the request is enough.
        with patch.object(service, "dispatch"):
            response = self.post(self.body(catalogue=self.catalogue.pk))
        self.assertEqual(response.status_code, 202)

    @override_settings(WHATSAPP_DEFAULT_CATALOGUE_ID=None)
    def test_the_catalogue_setting_decides_when_several_exist(self):
        Catalogue.objects.create(name="Wheat", status="Published")
        with override_settings(WHATSAPP_DEFAULT_CATALOGUE_ID=self.catalogue.pk):
            with patch.object(service, "dispatch"):
                self.assertEqual(self.post(self.body()).status_code, 202)


class ProcessingTests(GenericApiTestCase):
    """The worker: RAG once, delivery, and what happens when either fails."""

    def record(self, **over):
        fields = {"provider": InboundMessage.PROVIDER_GENERIC,
                  "message_id": "p1", "from_number": "+919876543210",
                  "message_type": "text", "text": "how much urea per acre?"}
        fields.update(over)
        return InboundMessage.objects.create(**fields)

    def test_the_existing_rag_pipeline_is_called_exactly_once(self):
        record = self.record()
        query = self.answered_query()

        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=query) as rag, \
             patch("queries.services.whatsapp.send_answer",
                   return_value={"ok": True}):
            service.answer_message(record.pk)

        rag.assert_called_once()
        kwargs = rag.call_args.kwargs
        self.assertEqual(kwargs["query_text"], "how much urea per acre?")
        self.assertEqual(kwargs["catalogue"], self.catalogue)
        self.assertEqual(kwargs["whatsapp_number"], "+919876543210")
        self.assertTrue(kwargs["generate_tts"])
        # No hook: the pipeline must not try to send on its own, because delivery
        # for this path is the service's job.
        self.assertIsNone(kwargs["hook"])

        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_ANSWERED)
        self.assertEqual(record.query, query)

    def test_the_hook_path_is_untouched_by_the_generic_one(self):
        # Same worker, different branch: a hook message still goes through
        # process_incoming_message, which still sends the text itself.
        hook = Hook.objects.create(name="H", whatsapp_number="+911",
                                   trigger_keyword="DHAN", catalogue=self.catalogue,
                                   status=Hook.STATUS_ACTIVE)
        record = self.record(provider=Hook.GATEWAY_PICKY, message_id="h1", hook=hook)

        with patch("queries.services.pipeline.process_incoming_message",
                   return_value=self.answered_query()) as hook_path, \
             patch("queries.services.pipeline.run_pipeline_for_text") as generic_path, \
             patch("queries.services.whatsapp.send_answer",
                   return_value={"ok": True}) as send:
            service.answer_message(record.pk)

        hook_path.assert_called_once()
        generic_path.assert_not_called()
        self.assertTrue(send.call_args.kwargs["text_already_sent"])

    def test_the_answer_is_delivered_through_the_existing_whatsapp_service(self):
        record = self.record()
        query = self.answered_query()

        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=query), \
             patch("queries.services.whatsapp.send_answer",
                   return_value={"ok": True}) as send:
            service.answer_message(record.pk)

        send.assert_called_once()
        args, kwargs = send.call_args
        self.assertIsNone(args[0])                 # no hook
        self.assertEqual(args[1], "+919876543210")
        self.assertEqual(args[2], query)
        # Nothing was sent before this, so the text is this call's job.
        self.assertFalse(kwargs["text_already_sent"])

    def test_the_language_the_caller_asked_for_is_used(self):
        record = self.record(language="Odia")
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=self.answered_query()) as rag, \
             patch("queries.services.whatsapp.send_answer", return_value={"ok": True}):
            service.answer_message(record.pk)
        self.assertEqual(rag.call_args.kwargs["farmer_language"], "Odia")

    def test_a_rag_failure_is_recorded_and_the_message_is_not_lost(self):
        record = self.record()
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   side_effect=RuntimeError("openai down")):
            service.answer_message(record.pk)

        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_FAILED)
        self.assertIn("openai down", record.error)
        self.assertEqual(record.attempts, 1)
        self.assertIsNone(record.query)

    def test_a_delivery_failure_keeps_the_answer(self):
        # The expensive half succeeded. Losing it because the gateway was down
        # would mean paying for it again.
        record = self.record()
        query = self.answered_query()
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=query), \
             patch("queries.services.whatsapp.send_answer",
                   side_effect=RuntimeError("gateway 502")):
            service.answer_message(record.pk)

        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_DELIVERY_FAILED)
        self.assertEqual(record.query, query)
        self.assertIn("gateway 502", record.error)

    def test_a_retry_after_a_delivery_failure_does_not_pay_for_the_answer_again(self):
        record = self.record(query=self.answered_query(),
                            status=InboundMessage.STATUS_DELIVERY_FAILED)

        with patch("queries.services.pipeline.run_pipeline_for_text") as rag, \
             patch("queries.services.whatsapp.send_answer",
                   return_value={"ok": True}) as send:
            service.answer_message(record.pk)

        rag.assert_not_called()
        send.assert_called_once()
        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_ANSWERED)

    def test_an_already_answered_message_is_never_answered_twice(self):
        record = self.record(query=self.answered_query(),
                            status=InboundMessage.STATUS_ANSWERED)
        with patch("queries.services.pipeline.run_pipeline_for_text") as rag, \
             patch("queries.services.whatsapp.send_answer") as send:
            service.answer_message(record.pk)
        rag.assert_not_called()
        send.assert_not_called()

    def test_a_reported_send_failure_is_treated_as_a_failure(self):
        # The sender reports {"ok": False} rather than raising when the gateway
        # rejects the message.
        record = self.record()
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=self.answered_query()), \
             patch("queries.services.whatsapp.send_answer",
                   return_value={"ok": False, "error": "401 from gateway"}):
            service.answer_message(record.pk)

        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_DELIVERY_FAILED)

    def test_a_tts_failure_does_not_lose_the_text_answer(self):
        # TTS is inside run_pipeline_for_text and already degrades to no audio;
        # this proves the text answer still goes out when there is no voice file.
        record = self.record()
        query = self.answered_query()          # no ai_response_voice_file
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=query), \
             patch("queries.services.whatsapp.send_via_picky_assist",
                   return_value={"ok": True}) as raw_send, \
             override_settings(PICKY_ASSIST_TOKEN="t", PUBLIC_BASE_URL=""):
            service.answer_message(record.pk)

        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_ANSWERED)
        # Text was sent; no audio url was invented for it.
        self.assertEqual(raw_send.call_args.kwargs.get("audio_url"), None)

    def test_a_message_with_no_recipient_is_a_delivery_failure_not_a_crash(self):
        record = self.record(from_number="")
        with patch("queries.services.pipeline.run_pipeline_for_text",
                   return_value=self.answered_query()):
            service.answer_message(record.pk)
        record.refresh_from_db()
        self.assertEqual(record.status, InboundMessage.STATUS_DELIVERY_FAILED)


class DispatchTests(GenericApiTestCase):

    def test_without_celery_the_thread_pool_is_used(self):
        # Celery is not installed in this project; dispatch must still work.
        with patch.object(service, "celery_available", return_value=False), \
             patch.object(service, "_pool") as pool:
            self.assertEqual(service.dispatch(1), "thread")
        pool.return_value.submit.assert_called_once_with(service.answer_message, 1)

    def test_with_celery_the_task_is_queued(self):
        fake_task = patch("hooks.tasks.answer_message_task", create=True).start()
        self.addCleanup(patch.stopall)
        with patch.object(service, "celery_available", return_value=True):
            self.assertEqual(service.dispatch(7), "celery")
        fake_task.delay.assert_called_once_with(7)

    def test_celery_is_only_used_when_a_broker_is_configured(self):
        with override_settings(CELERY_BROKER_URL=""):
            self.assertFalse(service.celery_available())


class HealthTests(GenericApiTestCase):

    def test_health_reports_each_dependency(self):
        response = self.client.get(HEALTH)
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "whatsapp")
        for field in ("api_configured", "celery_configured", "rag_configured",
                      "tts_configured", "stt_configured"):
            self.assertIn(field, body)
            self.assertIsInstance(body[field], bool)
        self.assertTrue(body["api_configured"])     # the key issued in setUp

    def test_health_never_returns_a_secret(self):
        secret = self.plaintext.split("_", 2)[2]
        with override_settings(OPENAI_API_KEY="sk-should-not-appear",
                               NARAKEET_KEY="narakeet-should-not-appear",
                               SARVAM_API_KEY="sarvam-should-not-appear",
                               PICKY_ASSIST_TOKEN="token-should-not-appear",
                               PICKY_ASSIST_WEBHOOK_SECRET="secret-should-not-appear"):
            body = self.client.get(HEALTH).content.decode()

        for leaked in (secret, "sk-should-not-appear", "narakeet-should-not-appear",
                       "sarvam-should-not-appear", "token-should-not-appear",
                       "secret-should-not-appear"):
            self.assertNotIn(leaked, body)

    def test_health_says_when_no_generic_key_exists(self):
        WhatsAppApiKey.objects.all().delete()
        self.assertFalse(self.client.get(HEALTH).json()["api_configured"])


class BackwardCompatibilityTests(GenericApiTestCase):

    def test_the_pickyassist_endpoint_is_still_routed(self):
        response = self.client.get("/api/v1/whatsapp/pickyassist/")
        self.assertEqual(response.status_code, 200)

    def test_the_legacy_hook_webhook_is_still_routed(self):
        response = self.client.get("/hooks/webhook/pickyassist/")
        self.assertEqual(response.status_code, 200)

    def test_a_generic_key_cannot_be_used_on_the_pickyassist_endpoint(self):
        response = self.client.post(
            "/api/v1/whatsapp/pickyassist/",
            data=json.dumps({"message_id": "x", "from": "+91987", "text": "hi"}),
            content_type="application/json", HTTP_X_API_KEY=self.plaintext)
        self.assertEqual(response.status_code, 401)


class LanguageTests(TestCase):

    def test_iso_codes_become_the_names_this_codebase_routes_on(self):
        # Sarvam is selected by language_hint == "Odia" and Narakeet by
        # language == "Odia", so the translation has to happen before either.
        for value, expected in (("or", "Odia"), ("od", "Odia"), ("ORY", "Odia"),
                                ("en", "English"), ("hi", "Hindi"),
                                ("Odia", "Odia"), ("English", "English")):
            with self.subTest(value=value):
                self.assertEqual(service.language_name(value), expected)

    def test_a_blank_language_uses_the_default_given(self):
        self.assertEqual(service.language_name("", "Odia"), "Odia")
        self.assertEqual(service.language_name(None, "Odia"), "Odia")

    def test_an_unknown_language_is_passed_through_rather_than_dropped(self):
        self.assertEqual(service.language_name("Marathi"), "Marathi")
