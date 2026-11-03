import logging
import os
import tempfile

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.views.decorators.http import require_POST
from django.views.generic import (
    CreateView, DeleteView, DetailView, ListView, TemplateView, UpdateView,
)

from core.mixins import RoleRequiredMixin

from .forms import CatalogueForm
from .models import Catalogue

log = logging.getLogger("irri")


class EditorMixin(RoleRequiredMixin):
    required_roles = ("Administrator", "Editor")


class CatalogueListView(LoginRequiredMixin, ListView):
    model = Catalogue
    template_name = "catalogues/list.html"
    context_object_name = "catalogues"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        qs = Catalogue.objects.all()
        ctx["total"] = qs.count()
        ctx["published"] = qs.filter(status="Published").count()
        ctx["ready"] = qs.filter(vector_db_status="Ready").count()
        ctx["linked"] = qs.filter(hooks__isnull=False).distinct().count()
        ctx["ready_catalogues"] = qs.filter(vector_db_status="Ready").order_by("name")
        # If a previous beta-test redirected here with ?result=<query_id>, show it inline.
        result_id = self.request.GET.get("result")
        if result_id:
            from queries.models import Query
            ctx["beta_result"] = Query.objects.filter(pk=result_id).select_related(
                "catalogue"
            ).prefetch_related("source_documents").first()
        return ctx


@login_required
@require_POST
def beta_test(request):
    """Run the real AI pipeline against a catalogue and redirect back to list with ?result=id.
    Accepts either a text query or an uploaded image (image-as-query)."""
    from queries.services.pipeline import run_pipeline_for_image, run_pipeline_for_text

    slug = request.POST.get("catalogue")
    language = request.POST.get("language") or "English"
    query_text = (request.POST.get("query_text") or "").strip()
    user_type = request.POST.get("user_type") or "farmer"
    image = request.FILES.get("image")
    if user_type not in ("farmer", "researcher"):
        user_type = "farmer"
    if not slug:
        messages.error(request, "Please select a catalogue.")
        return redirect("catalogues:list")
    if not query_text and not image:
        messages.error(request, "Please enter a query or upload an image.")
        return redirect("catalogues:list")
    catalogue = get_object_or_404(Catalogue, slug=slug)
    if catalogue.vector_db_status != Catalogue.VDB_READY:
        messages.error(request, f"Catalogue '{catalogue.name}' has no vector DB yet. Build it first.")
        return redirect("catalogues:list")
    try:
        if image:
            q = run_pipeline_for_image(
                catalogue=catalogue,
                farmer_language=language,
                uploaded_image=image,
                district="",
                generate_tts=False,
                user_type=user_type,
                user_query_text=query_text,
            )
        else:
            q = run_pipeline_for_text(
                catalogue=catalogue,
                farmer_language=language,
                query_text=query_text,
                district="",
                generate_tts=False,
                user_type=user_type,
            )
        return redirect(f"{reverse('catalogues:list')}?result={q.pk}")
    except Exception as exc:
        log.exception("Beta test failed")
        messages.error(request, f"Pipeline error: {exc}")
        return redirect("catalogues:list")


@login_required
@require_POST
def transcribe_audio(request):
    """
    Transcribe an uploaded audio blob to text in the source language.
    Used by the Beta Test mic button. POST multipart with:
        audio=<blob>
        language=<English|Hindi|Odia|Bengali|Telugu|Tamil>
    Returns JSON: {"ok": true, "text": "..."} or {"ok": false, "error": "..."}.
    """
    audio = request.FILES.get("audio")
    language = (request.POST.get("language") or "English").strip()
    if not audio:
        return JsonResponse({"ok": False, "error": "No audio file"}, status=400)

    suffix = os.path.splitext(audio.name)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in audio.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name
    try:
        from queries.services.pipeline import stt_transcribe
        text, stt_backend = stt_transcribe(tmp_path, language_hint=language)
        return JsonResponse({"ok": True, "text": text or ""})
    except Exception as exc:
        log.exception("Transcribe failed")
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


class CatalogueDetailView(LoginRequiredMixin, DetailView):
    model = Catalogue
    template_name = "catalogues/detail.html"
    context_object_name = "catalogue"

    def get_context_data(self, **kwargs):
        from django.core.paginator import Paginator
        from django.db.models import Count

        ctx = super().get_context_data(**kwargs)
        catalogue = self.object

        # Paginated documents
        docs_qs = catalogue.documents.all().select_related("category", "subcategory")
        docs_paginator = Paginator(docs_qs, 20)
        ctx["documents_page"] = docs_paginator.get_page(self.request.GET.get("docs_page"))

        # Paginated image groups (with image counts + cover thumbnail prefetch)
        groups_qs = (
            catalogue.image_groups
            .annotate(n_images=Count("images"))
            .prefetch_related("images")
            .order_by("prefix")
        )
        groups_paginator = Paginator(groups_qs, 12)
        ctx["image_groups_page"] = groups_paginator.get_page(self.request.GET.get("groups_page"))
        ctx["image_groups_total"] = groups_qs.count()
        return ctx


class CatalogueCreateView(EditorMixin, CreateView):
    model = Catalogue
    form_class = CatalogueForm
    template_name = "catalogues/form.html"

    def get_success_url(self):
        return reverse("catalogues:detail", kwargs={"slug": self.object.slug})


class CatalogueUpdateView(EditorMixin, UpdateView):
    model = Catalogue
    form_class = CatalogueForm
    template_name = "catalogues/form.html"

    def get_success_url(self):
        return reverse("catalogues:detail", kwargs={"slug": self.object.slug})


class CatalogueDeleteView(EditorMixin, DeleteView):
    model = Catalogue
    template_name = "catalogues/confirm_delete.html"
    success_url = reverse_lazy("catalogues:list")


@login_required
@require_POST
def build_vdb(request, slug):
    role = getattr(request.user, "role", None)
    if not role or role.name not in ("Administrator", "Editor"):
        return HttpResponseForbidden("Not permitted.")
    catalogue = get_object_or_404(Catalogue, slug=slug)
    from .services.vector_store import build_catalogue_vector_db

    try:
        stats = build_catalogue_vector_db(catalogue)
        messages.success(
            request,
            f"Vector DB built: {stats['chunks']} chunks from {stats['documents']} documents.",
        )
    except Exception as exc:
        log.exception("Vector DB build failed")
        messages.error(request, f"Build failed: {exc}")
    return redirect("catalogues:detail", slug=slug)


class CatalogueSearchView(LoginRequiredMixin, TemplateView):
    template_name = "catalogues/search.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        slug = kwargs["slug"]
        catalogue = get_object_or_404(Catalogue, slug=slug)
        ctx["catalogue"] = catalogue
        q = self.request.GET.get("q", "").strip()
        ctx["q"] = q
        ctx["results"] = []
        if q and catalogue.vector_db_status == Catalogue.VDB_READY:
            from .services.vector_store import search_catalogue

            try:
                ctx["results"] = search_catalogue(catalogue, q, n_results=6)
            except Exception as exc:
                log.exception("Search failed")
                ctx["error"] = str(exc)
        return ctx
