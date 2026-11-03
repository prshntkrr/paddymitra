# Demo Chat Terminal Feature

## Overview
This document describes the new Run Demo Query chat terminal feature added to the PaddyMitra Django app.

The new demo page now supports:
- WhatsApp-style chat terminal UI
- Text and microphone-based question input
- Language selection for query processing
- Persona selection (`farmer` / `researcher`)
- Optional OpenAI TTS response playback
- A recent query table with feedback controls

## Files Updated

- `templates/queries/demo_form.html`
  - Replaced the legacy demo text input with a chat terminal UI
  - Added a send button, mic recorder, live chat bubble rendering, and recent query table
  - Added JavaScript for AJAX chat submission, audio transcription, and response rendering

- `queries/views.py`
  - Added `demo_chat` AJAX endpoint (`POST /queries/demo/chat/`)
  - Added recent query loading in `QueryDemoView.get_context_data`
  - Updated `set_feedback` to support redirecting back to the demo page via `next`

- `queries/urls.py`
  - Added route: `path("demo/chat/", views.demo_chat, name="demo_chat")`

## Feature Behavior

1. User opens the demo page at `/queries/demo/`.
2. The page displays:
   - catalogue selector
   - farmer language selector
   - district selector
   - persona choice
   - chat window
   - text input
   - mic button
   - Send button
   - optional image upload
   - query history table with feedback dropdowns
3. User enters text or records audio.
4. The app sends the request to the new endpoint:
   - `POST /queries/demo/chat/`
   - payload includes `catalogue`, `farmer_language`, `user_type`, `generate_tts`, `query_text`, `district`
5. If audio is provided and query text is empty, the endpoint transcribes audio with `stt_transcribe()`.
6. The request runs `run_pipeline_for_text()` and saves a `Query` object.
7. The response returns text and optional audio URL.
8. The UI renders the question and the AI reply as chat bubbles.
9. Feedback can be provided in the saved queries table.

## Backend Behavior

- `demo_chat` supports both direct text and audio queries.
- The endpoint returns JSON containing:
  - `ok`
  - `query_id`
  - `original_query_text`
  - `original_query_language`
  - `ai_response_text_local`
  - `ai_response_text`
  - `ai_response_voice_url`
  - `created_at`
  - `feedback`
- Audio transcription is delegated to `queries.services.pipeline.stt_transcribe()`.
- The query is persisted as a `Query` model record, including response and feedback fields.

## Notes

- The chat UI currently uses an AJAX POST to the demo chat endpoint.
- TTS playback is only available when `generate_tts` is enabled.
- The query history table shows recent demo queries and lets the user rate them.

## Testing

- Ensure the server is running.
- Open `/queries/demo/` and verify the chat terminal appears.
- Type a question and click Send.
- Use the mic button to record voice and transcribe it into the chat input.
- Confirm responses show in the chat window and audio playback appears when TTS is generated.
- Submit feedback from the recent queries table.

## Future Improvements

- Auto-refresh the query history table after each successful chat exchange.
- Add a clear chat button.
- Display response status/typing indicator during pipeline execution.
- Make the chat panel persist previous session messages.
