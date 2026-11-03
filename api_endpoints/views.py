from __future__ import annotations

import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.views.generic import DetailView, ListView

from catalogues.models import Catalogue
from core.mixins import RoleRequiredMixin
from queries.models import Query

from .models import GeneratedAPI

log = logging.getLogger("irri")

EDITOR_ROLES = ("Administrator", "Editor")


# ----------------- Internal management UI -----------------

class APIListView(RoleRequiredMixin, ListView):
    model = GeneratedAPI
    template_name = "api_endpoints/list.html"
    context_object_name = "apis"
    paginate_by = 25
    required_roles = EDITOR_ROLES


class APIDetailView(RoleRequiredMixin, DetailView):
    model = GeneratedAPI
    template_name = "api_endpoints/detail.html"
    context_object_name = "api"
    slug_field = "public_id"
    slug_url_kwarg = "public_id"
    required_roles = EDITOR_ROLES

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["full_url"] = self.request.build_absolute_uri(self.object.relative_url)
        return ctx


@login_required
@require_POST
def create_api_from_query(request):
    """POST handler used by the 'Generate API' button on the Beta Test result panel."""
    role = getattr(request.user, "role", None)
    role_name = role.name if role else ""
    if role_name not in EDITOR_ROLES:
        messages.error(request, "Only Editors / Administrators can generate APIs.")
        return redirect("catalogues:list")

    query_id = request.POST.get("source_query_id")
    name = (request.POST.get("name") or "").strip()
    description = (request.POST.get("description") or "").strip()
    if not query_id:
        messages.error(request, "Missing source query.")
        return redirect("catalogues:list")
    q = get_object_or_404(Query, pk=query_id)
    if not q.catalogue:
        messages.error(request, "Source query has no catalogue.")
        return redirect("queries:detail", pk=q.pk)

    user_type = (q.pipeline_trace or {}).get("user_type") or "farmer"
    if not name:
        name = f"{q.catalogue.name} · {user_type.title()} · {q.original_query_language}"

    api = GeneratedAPI.objects.create(
        name=name[:200],
        description=description,
        catalogue=q.catalogue,
        language=q.original_query_language or "English",
        user_type=user_type,
        sample_query=q.original_query_text or q.translated_query_text,
        source_query=q,
        created_by=request.user,
    )
    messages.success(request, f"API generated: {api.name}")
    return redirect("api_endpoints:detail", public_id=api.public_id)


@login_required
@require_POST
def delete_api(request, public_id):
    role = getattr(request.user, "role", None)
    role_name = role.name if role else ""
    if role_name not in EDITOR_ROLES:
        messages.error(request, "Only Editors / Administrators can delete APIs.")
        return redirect("api_endpoints:list")
    api = get_object_or_404(GeneratedAPI, public_id=public_id)
    api.delete()
    messages.success(request, "API deleted.")
    return redirect("api_endpoints:list")


@login_required
@require_POST
def rotate_token(request, public_id):
    role = getattr(request.user, "role", None)
    role_name = role.name if role else ""
    if role_name not in EDITOR_ROLES:
        return redirect("api_endpoints:list")
    from .models import _new_token
    api = get_object_or_404(GeneratedAPI, public_id=public_id)
    api.token = _new_token()
    api.save(update_fields=["token", "updated_at"])
    messages.success(request, "Token rotated.")
    return redirect("api_endpoints:detail", public_id=api.public_id)


# ----------------- Public dispatch endpoint -----------------

def _extract_token(request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("token "):
        return auth.split(None, 1)[1].strip()
    return request.headers.get("X-API-Token", "") or request.GET.get("token", "")


@csrf_exempt
def dispatch_api(request, public_id):
    """
    Public endpoint. POST text or multipart with an image.

    Headers:
        Authorization: Token <api-token>
    Body:
        application/json:  {"query": "your question here"}
        multipart/form-data: image=<file>  (optional 'query' field)
    Returns JSON:
        {
          "ok": true,
          "query_id": <int>,
          "answer_en": "...",
          "answer_local": "...",
          "language": "...",
          "matched_image_url": "..." | null,
          "response_time_seconds": ...
        }
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST only"}, status=405)

    api = GeneratedAPI.objects.filter(public_id=public_id).first()
    if not api or not api.is_active:
        return JsonResponse({"ok": False, "error": "API not found or inactive"}, status=404)

    token = _extract_token(request)
    if not token or token != api.token:
        return JsonResponse({"ok": False, "error": "Invalid or missing token"}, status=401)

    # Accept JSON or multipart.
    image = request.FILES.get("image")
    audio = request.FILES.get("audio")
    query_text = ""
    language = api.language
    if request.content_type and request.content_type.startswith("application/json"):
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except Exception:
            return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)
        query_text = (payload.get("query") or "").strip()
        language = (payload.get("language") or language).strip() or api.language
    else:
        query_text = (request.POST.get("query") or "").strip()
        language = (request.POST.get("language") or language).strip() or api.language

    # Optional audio: transcribe to source-language text, then continue.
    if audio is not None:
        import os, tempfile
        suffix = os.path.splitext(audio.name)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in audio.chunks():
                tmp.write(chunk)
            audio_path = tmp.name
        try:
            from queries.services.pipeline import stt_transcribe
            transcribed, _stt_backend = stt_transcribe(audio_path, language_hint=language)
            if transcribed:
                # If the caller also typed a query, combine them.
                query_text = (query_text + " " + transcribed).strip() if query_text else transcribed
        except Exception as exc:
            log.exception("Audio transcription failed")
            return JsonResponse({"ok": False, "error": f"Audio transcription failed: {exc}"}, status=500)
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

    if not query_text and not image:
        return JsonResponse(
            {"ok": False, "error": "Provide 'query' text, 'audio' file, or 'image' file"},
            status=400,
        )

    from queries.services.pipeline import run_pipeline_for_image, run_pipeline_for_text

    try:
        if image:
            q = run_pipeline_for_image(
                catalogue=api.catalogue,
                farmer_language=language,
                uploaded_image=image,
                user_type=api.user_type,
                user_query_text=query_text,  # may be empty -> pure image query
            )
        else:
            q = run_pipeline_for_text(
                catalogue=api.catalogue,
                farmer_language=language,
                query_text=query_text,
                user_type=api.user_type,
            )
    except Exception as exc:
        log.exception("Generated API dispatch failed")
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)

    api.request_count = (api.request_count or 0) + 1
    api.last_called_at = timezone.now()
    api.save(update_fields=["request_count", "last_called_at", "updated_at"])

    matched_url = None
    trace = q.pipeline_trace or {}
    iq = trace.get("image_query") if isinstance(trace, dict) else None
    if iq:
        matched_url = iq.get("matched_image_url")

    return JsonResponse({
        "ok": True,
        "query_id": q.pk,
        "answer_en": q.ai_response_text,
        "answer_local": q.ai_response_text_local,
        "language": language,
        "transcribed_query": query_text if audio is not None else None,
        "matched_image_url": matched_url,
        "response_time_seconds": q.response_time_seconds,
    })
