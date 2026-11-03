from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import (
    CreateView, DeleteView, DetailView, ListView, UpdateView,
)

from core.mixins import RoleRequiredMixin

from .forms import KeywordForm
from .models import Keyword


class EditorMixin(RoleRequiredMixin):
    required_roles = ("Administrator", "Editor")


class KeywordListView(LoginRequiredMixin, ListView):
    model = Keyword
    template_name = "keywords/list.html"
    paginate_by = 25
    context_object_name = "keywords"

    def get_queryset(self):
        qs = Keyword.objects.all()
        q = self.request.GET
        if q.get("parent"):
            qs = qs.filter(parent_category=q["parent"])
        if q.get("q"):
            qs = qs.filter(keyword__icontains=q["q"])
        return qs.order_by("keyword")

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        qs = Keyword.objects.all()
        ctx["total"] = qs.count()
        ctx["published"] = qs.filter(status="Published").count()
        ctx["draft"] = qs.filter(status="Draft").count()
        ctx["disabled"] = qs.filter(status="Disabled").count()
        ctx["parent_choices"] = Keyword.PARENT_CHOICES
        return ctx


class KeywordDetailView(LoginRequiredMixin, DetailView):
    model = Keyword
    template_name = "keywords/detail.html"
    context_object_name = "kw"


class KeywordCreateView(EditorMixin, CreateView):
    model = Keyword
    form_class = KeywordForm
    template_name = "keywords/form.html"
    success_url = reverse_lazy("keywords:list")


class KeywordUpdateView(EditorMixin, UpdateView):
    model = Keyword
    form_class = KeywordForm
    template_name = "keywords/form.html"
    success_url = reverse_lazy("keywords:list")


class KeywordDeleteView(EditorMixin, DeleteView):
    model = Keyword
    template_name = "keywords/confirm_delete.html"
    success_url = reverse_lazy("keywords:list")
