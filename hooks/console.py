"""
hooks/console.py
================
The WhatsApp API, on the screen where someone goes looking for an API.

WHY THIS EXISTS
---------------
Everything needed to run the WhatsApp integration already existed — several keys
per hook, rotation with a grace period, revocation — but only as management
commands. Anyone who did not have shell access to the server could not tell that
the WhatsApp API existed at all: the API section listed only the generated
per-catalogue APIs, and the keys were invisible.

WHAT A KEY IS, AND WHY THE PAGE LOOKS LIKE THIS
-----------------------------------------------
Only a prefix and a SHA-256 hash are stored, so the plaintext exists exactly
once: in the response to the request that created it. The page therefore shows
it in a banner, says plainly that it will not be shown again, and never tries to
display it afterwards.

ROTATION IS NOT REVOCATION. Rotating issues a replacement and leaves the old key
working for a grace period, because changing a credential means editing the
configuration at the gateway and anything a farmer sends between the two edits
would otherwise be refused — an outage nobody would connect to the rotation.
Revoking kills a key immediately, which is what you want when it has leaked.
"""
from __future__ import annotations

import logging

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .models import Hook, InboundMessage, WhatsAppApiKey

log = logging.getLogger("irri")

EDITOR_ROLES = ("Administrator", "Editor")


def _may_edit(request) -> bool:
    role = getattr(request.user, "role", None)
    return bool(role and role.name in EDITOR_ROLES)


@login_required
def whatsapp_api(request):
    """The endpoints, their readiness, and every key that can call them."""
    keys = list(WhatsAppApiKey.objects.select_related("hook")
                .order_by("revoked_at", "-created_at"))
    show_revoked = request.GET.get("revoked") == "1"
    if not show_revoked:
        keys = [k for k in keys if k.revoked_at is None]

    base = request.build_absolute_uri("/").rstrip("/")
    return render(request, "api_endpoints/whatsapp.html", {
        "keys": keys,
        "show_revoked": show_revoked,
        "hooks": Hook.objects.select_related("catalogue").order_by("trigger_keyword"),
        "can_edit": _may_edit(request),
        "endpoints": [
            ("Inbound webhook", "POST", f"{base}/api/v1/whatsapp/pickyassist/",
             "Where the gateway posts a farmer's message. The hook is chosen by "
             "the key, the keyword, or the number it was sent to."),
            ("Generic message", "POST", f"{base}/api/v1/whatsapp/message/",
             "Provider-independent. No hook: the catalogue is chosen "
             "automatically from the question."),
            ("Health", "GET", f"{base}/api/v1/whatsapp/health/",
             "Whether this deployment can answer right now. Reports what is set, "
             "never a value."),
        ],
        # The same three prerequisites the health endpoint reports, because this
        # is where someone stands when they are wondering why nothing arrives.
        "outbound_token_set": bool(settings.PICKY_ASSIST_TOKEN),
        "application_set": bool(getattr(settings, "PICKY_ASSIST_APPLICATION", "")),
        "public_base_url": getattr(settings, "PUBLIC_BASE_URL", ""),
        "webhook_secret_set": bool(settings.PICKY_ASSIST_WEBHOOK_SECRET),
        "unauthenticated_allowed":
            bool(getattr(settings, "WHATSAPP_ALLOW_UNAUTHENTICATED_WEBHOOK", False)),
        "messages_waiting": InboundMessage.objects.exclude(
            status=InboundMessage.STATUS_ANSWERED).count(),
        "new_key": request.session.pop("whatsapp_new_key", None),
    })


def _remember_once(request, plaintext: str, note: str) -> None:
    """Carry the plaintext through the redirect, once, then forget it.

    In the session rather than the URL: a key in a query string ends up in the
    browser history, the access log and anything sitting in front of the app.
    """
    request.session["whatsapp_new_key"] = {"key": plaintext, "note": note}


@login_required
@require_POST
def create_whatsapp_key(request):
    """Issue a key — for one hook, or generic. As many as are wanted."""
    if not _may_edit(request):
        messages.error(request, "Only Editors / Administrators can create API keys.")
        return redirect("api_endpoints:whatsapp")

    name = (request.POST.get("name") or "").strip()
    hook_id = (request.POST.get("hook") or "").strip()
    hook = None
    if hook_id:
        hook = get_object_or_404(Hook, pk=hook_id)
    key, plaintext = WhatsAppApiKey.issue(hook=hook, name=name,
                                          created_by=request.user)
    log.info("whatsapp api key created: %s scope=%s by=%s",
             key.masked, key.scope, request.user)
    _remember_once(request, plaintext,
                   f"New key for {key.scope}. Copy it now — it is stored only "
                   f"as a hash and cannot be shown again.")
    messages.success(request, "API key created.")
    return redirect("api_endpoints:whatsapp")


@login_required
@require_POST
def rotate_whatsapp_key(request, pk):
    """Replace a key, leaving the old one usable for its grace period."""
    if not _may_edit(request):
        messages.error(request, "Only Editors / Administrators can rotate API keys.")
        return redirect("api_endpoints:whatsapp")

    old = get_object_or_404(WhatsAppApiKey, pk=pk)
    raw_hours = (request.POST.get("grace_hours") or "").strip()
    grace = int(raw_hours) if raw_hours.isdigit() else None
    new, plaintext = old.rotate(grace_hours=grace)
    log.info("whatsapp api key rotated: %s -> %s by=%s",
             old.masked, new.masked, request.user)
    _remember_once(request, plaintext,
                   f"Replacement for {old.masked}. The old key keeps working "
                   f"until {old.expires_at:%d %b %Y, %H:%M} so you can update "
                   f"the gateway without dropping a message.")
    messages.success(request, "API key rotated.")
    return redirect("api_endpoints:whatsapp")


@login_required
@require_POST
def revoke_whatsapp_key(request, pk):
    """Kill a key now. For a key that has leaked, not for routine replacement."""
    if not _may_edit(request):
        messages.error(request, "Only Editors / Administrators can revoke API keys.")
        return redirect("api_endpoints:whatsapp")

    key = get_object_or_404(WhatsAppApiKey, pk=pk)
    key.revoke()
    log.warning("whatsapp api key revoked: %s by=%s", key.masked, request.user)
    messages.success(request, f"{key.masked} revoked. Anything using it will "
                              f"now be refused.")
    return redirect("api_endpoints:whatsapp")
