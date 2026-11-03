"""Odia speech through Sarvam — with no key in the source and no silent failures.

Every test here mocks the SDK. Nothing in this file reaches the network, and no
test needs a real subscription key.

Two things are being protected. The first is the CALLERS: `pipeline.tts_synthesize`
calls `synthesize(cleaned_text)` with one positional argument, and that has to go
on working. The second is the FAILURE MODE: a farmer who gets no audio still has
the written answer, but a farmer who gets half a second of silence believes the
system spoke and heard nothing — so an unusable response must raise rather than
return empty bytes.
"""
import base64
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from queries.services import narakeet_tts
from queries.services.narakeet_tts import (SarvamNotConfigured, SarvamTTSError,
                                           synthesize)

AUDIO = b"\x49\x44\x33fake-mp3-bytes"
ENCODED = base64.b64encode(AUDIO).decode()
KEY = "test-key-not-a-real-one"


def fake_client(audios=(ENCODED,), raises=None):
    """A stand-in for SarvamAI whose convert() records how it was called."""
    client = MagicMock()
    if raises is not None:
        client.text_to_speech.convert.side_effect = raises
    else:
        response = MagicMock()
        response.audios = list(audios)
        client.text_to_speech.convert.return_value = response
    return client


class SarvamTTSTestCase(SimpleTestCase):
    """Shared plumbing: a stubbed client, and no cached client between tests."""

    def setUp(self):
        narakeet_tts.reset_client_cache()
        self.addCleanup(narakeet_tts.reset_client_cache)
        self.client = fake_client()
        patcher = patch.object(narakeet_tts, "_new_client",
                               side_effect=lambda key: self.client)
        self.new_client = patcher.start()
        self.addCleanup(patcher.stop)

    def convert_kwargs(self):
        return self.client.text_to_speech.convert.call_args.kwargs


@override_settings(SARVAM_API_KEY=KEY)
class SynthesisTests(SarvamTTSTestCase):
    """The happy path, and the call the SDK is actually given."""

    def test_the_audio_comes_back_decoded(self):
        self.assertEqual(AUDIO, synthesize("ଧାନ ଚାଷ"))

    def test_one_positional_argument_still_works(self):
        # The contract pipeline.tts_synthesize relies on.
        self.assertEqual(AUDIO, synthesize("ଧାନ ଚାଷ"))
        self.assertEqual(1, self.client.text_to_speech.convert.call_count)

    def test_the_documented_defaults_are_sent(self):
        synthesize("ଧାନ ଚାଷ")
        sent = self.convert_kwargs()

        self.assertEqual("bulbul:v3", sent["model"])
        self.assertEqual("shubh", sent["speaker"])
        self.assertEqual("od-IN", sent["target_language_code"])
        self.assertEqual("ଧାନ ଚାଷ", sent["text"])
        self.assertEqual(0.9, sent["pace"])

    def test_only_parameters_the_installed_sdk_declares_are_sent(self):
        # Guards against inventing a parameter the SDK would reject. Checked
        # against the real signature, not a list written from memory.
        import inspect

        from sarvamai.text_to_speech.client import TextToSpeechClient

        allowed = set(inspect.signature(TextToSpeechClient.convert).parameters)
        synthesize("ଧାନ ଚାଷ")

        self.assertTrue(set(self.convert_kwargs()) <= allowed,
                        set(self.convert_kwargs()) - allowed)

    def test_the_voice_and_language_can_be_overridden(self):
        synthesize("hello", voice="anushka", target_language_code="hi-IN")
        sent = self.convert_kwargs()

        self.assertEqual("anushka", sent["speaker"])
        self.assertEqual("hi-IN", sent["target_language_code"])

    def test_the_model_and_pace_can_be_overridden(self):
        synthesize("hello", model="bulbul:v2", pace=1.1)
        sent = self.convert_kwargs()

        self.assertEqual("bulbul:v2", sent["model"])
        self.assertEqual(1.1, sent["pace"])

    def test_the_text_is_trimmed_before_it_is_spoken(self):
        synthesize("  ଧାନ  ")

        self.assertEqual("ଧାନ", self.convert_kwargs()["text"])

    def test_the_client_is_built_once_and_reused(self):
        synthesize("one")
        synthesize("two")

        self.assertEqual(1, self.new_client.call_count)

    def test_settings_can_change_the_voice_without_a_code_change(self):
        with override_settings(SARVAM_TTS_SPEAKER="vidya", SARVAM_TTS_MODEL="bulbul:v2",
                               SARVAM_TTS_LANGUAGE_CODE="bn-IN", SARVAM_TTS_PACE=1.2):
            synthesize("hello", voice="", target_language_code="")
        sent = self.convert_kwargs()

        self.assertEqual("vidya", sent["speaker"])
        self.assertEqual("bulbul:v2", sent["model"])
        self.assertEqual("bn-IN", sent["target_language_code"])
        self.assertEqual(1.2, sent["pace"])


@override_settings(SARVAM_API_KEY=KEY)
class EmptyTextTests(SarvamTTSTestCase):
    """Nothing to say is a caller error, not a request to send."""

    def test_empty_and_blank_text_is_refused(self):
        for value in ("", "   ", "\n", None, 0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    synthesize(value)

        self.client.text_to_speech.convert.assert_not_called()


class ApiKeyTests(SarvamTTSTestCase):
    """Where the key comes from, and what happens when it is absent."""

    @override_settings(SARVAM_API_KEY=KEY)
    def test_the_key_comes_from_settings(self):
        synthesize("ଧାନ")

        self.new_client.assert_called_once_with(KEY)

    @override_settings(SARVAM_API_KEY="")
    def test_the_environment_is_used_when_settings_has_none(self):
        with patch.dict("os.environ", {"SARVAM_API_KEY": "env-key"}):
            synthesize("ଧାନ")

        self.new_client.assert_called_once_with("env-key")

    @override_settings(SARVAM_API_KEY="")
    def test_no_key_anywhere_raises_and_sends_nothing(self):
        with patch.dict("os.environ", {"SARVAM_API_KEY": ""}):
            with self.assertRaises(SarvamNotConfigured) as caught:
                synthesize("ଧାନ")

        self.assertIn("SARVAM_API_KEY", str(caught.exception))
        self.client.text_to_speech.convert.assert_not_called()

    def test_no_key_is_written_into_the_source(self):
        # The reason this module was rewritten: a live subscription key used to
        # sit in it as a literal.
        import pathlib

        source = pathlib.Path(narakeet_tts.__file__).read_text(encoding="utf-8")

        self.assertNotIn("api_subscription_key=\"sk_", source)
        self.assertNotIn("sk_", source)


@override_settings(SARVAM_API_KEY=KEY)
class FailureTests(SarvamTTSTestCase):
    """No silence, ever — and no secret in the error."""

    def test_an_api_failure_raises_rather_than_returning_nothing(self):
        self.client = fake_client(raises=RuntimeError("503 Service Unavailable"))

        with self.assertRaises(SarvamTTSError) as caught:
            synthesize("ଧାନ")

        self.assertIn("503", str(caught.exception))

    def test_the_key_never_appears_in_the_error(self):
        self.client = fake_client(raises=RuntimeError(f"401 for key {KEY}"))

        with self.assertRaises(SarvamTTSError) as caught:
            synthesize("ଧାନ")

        self.assertNotIn(KEY, str(caught.exception))
        self.assertIn("***", str(caught.exception))

    def test_a_response_with_no_audio_raises(self):
        self.client = fake_client(audios=())

        with self.assertRaises(SarvamTTSError):
            synthesize("ଧାନ")

    def test_audio_that_is_not_base64_raises(self):
        self.client = fake_client(audios=("not base64 !!!",))

        with self.assertRaises(SarvamTTSError):
            synthesize("ଧାନ")

    def test_an_empty_audio_track_raises(self):
        self.client = fake_client(audios=("",))

        with self.assertRaises(SarvamTTSError):
            synthesize("ଧାନ")


@override_settings(SARVAM_API_KEY=KEY)
class PipelineIntegrationTests(SarvamTTSTestCase):
    """The caller's side of the contract."""

    def test_odia_goes_through_sarvam(self):
        # The text here is Latin on purpose: tts_synthesize routes on the
        # `language` argument, and it also prints the outgoing text to stdout,
        # which raises UnicodeEncodeError on a cp1252 Windows console. That
        # print is pre-existing and outside this change; the routing is what is
        # under test.
        from queries.services import pipeline

        with patch.object(pipeline, "prepare_for_tts", side_effect=lambda t, lang: t):
            audio = pipeline.tts_synthesize("dhana chasa", language="Odia")

        self.assertEqual(AUDIO, audio)
        self.assertEqual("od-IN", self.convert_kwargs()["target_language_code"])
        self.assertEqual("bulbul:v3", self.convert_kwargs()["model"])

    def test_other_languages_still_use_openai(self):
        from queries.services import pipeline

        with patch.object(pipeline, "_openai_tts_synthesize",
                          return_value=b"openai") as openai:
            audio = pipeline.tts_synthesize("hello", language="English")

        self.assertEqual(b"openai", audio)
        openai.assert_called_once()
        self.client.text_to_speech.convert.assert_not_called()
