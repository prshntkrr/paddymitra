import logging

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from django.views.generic import (
    CreateView, DeleteView, DetailView, ListView, UpdateView,
)

from core.mixins import RoleRequiredMixin

from . import inbound
from .forms import HookForm
from .models import Hook, InboundMessage, WhatsAppApiKey

log = logging.getLogger("irri")


class AdminMixin(RoleRequiredMixin):
    required_roles = ("Administrator",)


class HookListView(LoginRequiredMixin, ListView):
    model = Hook
    template_name = "hooks/list.html"
    context_object_name = "hooks"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        qs = Hook.objects.all()
        ctx["total"] = qs.count()
        ctx["active"] = qs.filter(status="Active").count()
        ctx["paused"] = qs.filter(status="Paused").count()
        ctx["messages_processed"] = sum(h.messages_processed_count for h in qs) or 0
        return ctx


class HookDetailView(LoginRequiredMixin, DetailView):
    model = Hook
    template_name = "hooks/detail.html"
    context_object_name = "hook"


class HookCreateView(AdminMixin, CreateView):
    model = Hook
    form_class = HookForm
    template_name = "hooks/form.html"
    success_url = reverse_lazy("hooks:list")


class HookUpdateView(AdminMixin, UpdateView):
    model = Hook
    form_class = HookForm
    template_name = "hooks/form.html"
    success_url = reverse_lazy("hooks:list")


class HookDeleteView(AdminMixin, DeleteView):
    model = Hook
    template_name = "hooks/confirm_delete.html"
    success_url = reverse_lazy("hooks:list")


@login_required
@require_POST
def toggle_hook(request, pk):
    role = getattr(request.user, "role", None)
    if not role or role.name != "Administrator":
        return HttpResponseForbidden("Not permitted.")
    hook = get_object_or_404(Hook, pk=pk)
    hook.status = "Paused" if hook.status == "Active" else "Active"
    hook.save()
    messages.info(request, f"Hook '{hook.name}' is now {hook.status}.")
    return redirect("hooks:list")


@method_decorator(csrf_exempt, name="dispatch")
class PickyAssistWebhookView(View):
    """WhatsApp inbound webhook. POST here from the gateway.

        POST /api/v1/whatsapp/pickyassist/
        X-Webhook-Secret: <PICKY_ASSIST_WEBHOOK_SECRET>
        Content-Type: application/json

        {
          "message_id": "abc123",            # any of: id, msg_id, wamid …
          "from": "+919876543210",           # any of: sender, mobile, wa_id …
          "to": "+911234567890",             # the business number (optional)
          "message_type": "text" | "audio",  # inferred when absent
          "text": "DHAN how much urea per acre?",
          "audio_url": "https://…/voice.ogg",
          "trigger_keyword": "DHAN",         # optional — the first word works too
          "district": "Bargarh"              # optional
        }

    Answers 200 as soon as the message is recorded; the farmer's answer is
    produced on a worker thread and pushed back through the gateway. Statuses:

        accepted   — recorded, the pipeline is running, the reply will follow
        duplicate  — the gateway is retrying a message already accepted
        ignored    — no text/audio, or no active hook claimed it

    A 401 means the shared secret did not match, and a 400 means the body was not
    readable. Neither is worth the gateway retrying.
    """

    def post(self, request, *args, **kwargs):
        body, status = inbound.receive(request, provider=Hook.GATEWAY_PICKY)
        return JsonResponse(body, status=status)

    def get(self, request, *args, **kwargs):
        # Gateways (and humans) check the URL is live before configuring it.
        return JsonResponse({"ok": True, "endpoint": "whatsapp-inbound",
                             "method": "POST"})


@csrf_exempt
@require_GET
def whatsapp_health(request):
    """Is this deployment able to answer a WhatsApp message right now?

    Everything that has to be true, in one place, because each of these was
    diagnosed the hard way at least once: no token means answers are generated
    and silently never delivered; no secret means the webhook refuses every
    message; no active hook means every message is ignored; no PUBLIC_BASE_URL
    means voice replies are skipped. Reports whether each is SET, never its value.
    """
    hooks = Hook.objects.filter(status=Hook.STATUS_ACTIVE)
    recent = InboundMessage.objects.order_by("-created_at")[:200]
    counts = {}
    for message in recent:
        counts[message.status] = counts.get(message.status, 0) + 1
    keys = list(WhatsAppApiKey.objects.select_related("hook")
                .filter(revoked_at__isnull=True))
    usable = [k for k in keys if k.is_usable]
    # The generic API's own readiness, merged in so this one URL answers for both
    # surfaces. Additive: every key the hook integration reported before is still
    # here, because a health check that changes shape breaks whatever watches it.
    from .api import health_fields
    return JsonResponse({
        **health_fields(),
        "ok": bool(settings.PICKY_ASSIST_TOKEN) and hooks.exists()
              and bool(usable or settings.PICKY_ASSIST_WEBHOOK_SECRET),
        "inbound_endpoint": "/api/v1/whatsapp/pickyassist/",
        "outbound_token_configured": bool(settings.PICKY_ASSIST_TOKEN),
        "webhook_secret_configured": bool(settings.PICKY_ASSIST_WEBHOOK_SECRET),
        # Prefixes only — the secret half is not stored, so there is nothing
        # here that could be used to call the endpoint.
        "api_keys": [
            {"key": k.masked, "hook": k.scope, "state": k.state,
             "name": k.name, "requests": k.request_count,
             "expires_at": k.expires_at.isoformat() if k.expires_at else None}
            for k in keys
        ],
        "usable_api_keys": len(usable),
        "unauthenticated_webhook_allowed":
            bool(settings.WHATSAPP_ALLOW_UNAUTHENTICATED_WEBHOOK),
        "voice_replies": bool(settings.PUBLIC_BASE_URL
                              and settings.WHATSAPP_SEND_VOICE),
        "workers": settings.WHATSAPP_WORKERS,
        "active_hooks": [
            {"keyword": h.trigger_keyword, "name": h.name,
             "number": h.whatsapp_number, "catalogue": h.catalogue_id,
             "language": h.primary_language}
            for h in hooks
        ],
        "last_200_messages_by_status": counts,
    })
