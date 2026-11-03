from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.views.generic import (
    CreateView, DeleteView, DetailView, ListView, UpdateView,
)

from core.mixins import RoleRequiredMixin

from .forms import DocumentForm
from .models import Document


def _is_admin_or_editor(user):
    role = getattr(user, "role", None)
    return role and role.name in ("Administrator", "Editor")


def _is_reviewer_or_admin(user):
    role = getattr(user, "role", None)
    return role and role.name in ("Administrator", "Reviewer")


class DocumentListView(LoginRequiredMixin, ListView):
    model = Document
    template_name = "documents/list.html"
    paginate_by = 25
    context_object_name = "documents"

    def get_queryset(self):
        qs = Document.objects.select_related("category", "subcategory").prefetch_related("keywords")
        q = self.request.GET
        if q.get("status"):
            qs = qs.filter(status=q["status"])
        if q.get("crop"):
            qs = qs.filter(crop=q["crop"])
        if q.get("content_type"):
            qs = qs.filter(content_type=q["content_type"])
        if q.get("q"):
            qs = qs.filter(title__icontains=q["q"])
        return qs.order_by("-uploaded_at")

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        qs = Document.objects.all()
        ctx["total"] = qs.count()
        ctx["published"] = qs.filter(status="Published").count()
        ctx["draft"] = qs.filter(status="Draft").count()
        ctx["pending_review"] = qs.filter(status="PendingReview").count()
        ctx["unpublished"] = qs.filter(status="Unpublished").count()
        ctx["crops"] = Document.CROP_CHOICES
        ctx["content_types"] = Document.CONTENT_TYPE_CHOICES
        ctx["statuses"] = Document.STATUS_CHOICES
        ctx["is_editor"] = _is_admin_or_editor(self.request.user)
        return ctx


class DocumentDetailView(LoginRequiredMixin, DetailView):
    model = Document
    template_name = "documents/detail.html"
    context_object_name = "doc"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        doc = self.object
        # Build the same content preview that the embedder sees.
        try:
            from catalogues.services.chunking import document_text
            preview = document_text(doc)
        except Exception:
            preview = doc.text_for_embedding()
        ctx["embedding_preview"] = preview or "(No content available — upload a file to generate embeddings from the PDF.)"
        return ctx


class DocumentCreateView(LoginRequiredMixin, CreateView):
    model = Document
    form_class = DocumentForm
    template_name = "documents/form.html"
    success_url = reverse_lazy("documents:list")

    def form_valid(self, form):
        form.instance.uploaded_by = self.request.user
        role = getattr(self.request.user, "role", None)
        role_name = role.name if role else ""
        # KW: force Draft; Editor/Admin may publish via separate button flow.
        if role_name == "Knowledge Worker":
            form.instance.status = Document.STATUS_DRAFT
        else:
            form.instance.status = form.instance.status or Document.STATUS_DRAFT
        messages.success(self.request, "Document created.")
        return super().form_valid(form)


class DocumentUpdateView(LoginRequiredMixin, UpdateView):
    model = Document
    form_class = DocumentForm
    template_name = "documents/form.html"

    def get_success_url(self):
        return reverse_lazy("documents:detail", kwargs={"pk": self.object.pk})

    def dispatch(self, request, *args, **kwargs):
        doc = self.get_object()
        role = getattr(request.user, "role", None)
        role_name = role.name if role else ""
        if role_name == "Knowledge Worker" and doc.uploaded_by_id != request.user.id:
            return HttpResponseForbidden("You can only edit documents you uploaded.")
        if role_name == "Reviewer":
            return HttpResponseForbidden("Reviewers cannot edit document content.")
        return super().dispatch(request, *args, **kwargs)


class DocumentDeleteView(RoleRequiredMixin, DeleteView):
    required_roles = ("Administrator",)
    model = Document
    template_name = "documents/confirm_delete.html"
    success_url = reverse_lazy("documents:list")


class ReviewQueueView(RoleRequiredMixin, ListView):
    required_roles = ("Administrator", "Reviewer")
    model = Document
    template_name = "documents/review_queue.html"
    context_object_name = "documents"

    def get_queryset(self):
        return Document.objects.filter(status=Document.STATUS_PENDING).order_by("-uploaded_at")


@login_required
@require_POST
def publish_document(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    if not _is_admin_or_editor(request.user):
        return HttpResponseForbidden("Not permitted.")
    doc.publish(by=request.user)
    messages.success(request, f"Document '{doc.title}' published.")
    return redirect("documents:list")


@login_required
@require_POST
def unpublish_document(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    if not _is_admin_or_editor(request.user):
        return HttpResponseForbidden("Not permitted.")
    doc.unpublish()
    messages.warning(request, f"Document '{doc.title}' unpublished.")
    return redirect("documents:list")


@login_required
@require_POST
def submit_for_review(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    if doc.uploaded_by_id != request.user.id and not _is_admin_or_editor(request.user):
        return HttpResponseForbidden("Not permitted.")
    doc.status = Document.STATUS_PENDING
    doc.save()
    messages.info(request, "Submitted for reviewer approval.")
    return redirect("documents:detail", pk=doc.pk)


@login_required
@require_POST
def approve_document(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    if not _is_reviewer_or_admin(request.user):
        return HttpResponseForbidden("Not permitted.")
    doc.status = Document.STATUS_PUBLISHED
    doc.published_on = timezone.now()
    doc.reviewed_by = request.user
    doc.save()
    messages.success(request, f"Approved: '{doc.title}'.")
    return redirect("documents:review_queue")


@login_required
@require_POST
def reject_document(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    if not _is_reviewer_or_admin(request.user):
        return HttpResponseForbidden("Not permitted.")
    notes = request.POST.get("notes", "")
    doc.status = Document.STATUS_DRAFT
    doc.review_notes = (doc.review_notes + "\n\n" if doc.review_notes else "") + notes
    doc.reviewed_by = request.user
    doc.save()
    messages.warning(request, f"Rejected: '{doc.title}'.")
    return redirect("documents:review_queue")
