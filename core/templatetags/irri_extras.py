from django import template
from django.utils.safestring import mark_safe

register = template.Library()


@register.simple_tag
def icon(name, size=16, cls=""):
    """Render an inline SVG icon that references a symbol from templates/partials/icons.html."""
    safe_name = str(name).replace('"', "").replace("'", "")
    safe_cls = str(cls).replace('"', "").replace("'", "")
    try:
        size_int = int(size)
    except (TypeError, ValueError):
        size_int = 16
    return mark_safe(
        f'<svg class="ico {safe_cls}" width="{size_int}" height="{size_int}" '
        f'aria-hidden="true" focusable="false" style="vertical-align:-0.18em;">'
        f'<use xlink:href="#icon-{safe_name}" href="#icon-{safe_name}"/></svg>'
    )


@register.simple_tag
def active(current, *targets):
    """Return 'active' class if current matches any target (for sidebar highlighting)."""
    return "active" if current in targets else ""


@register.filter
def role_has(user, required):
    """Usage: {% if user|role_has:"Administrator,Editor" %}"""
    if not user or not user.is_authenticated:
        return False
    role = getattr(user, "role", None)
    if not role:
        return False
    return role.name in [r.strip() for r in required.split(",")]


@register.filter
def badge_class(value):
    """Map status values to CSS badge classes."""
    mapping = {
        "Published": "badge-green",
        "Draft": "badge-amber",
        "PendingReview": "badge-amber",
        "Pending Review": "badge-amber",
        "Unpublished": "badge-gray",
        "Disabled": "badge-red",
        "Active": "badge-green",
        "Inactive": "badge-gray",
        "Paused": "badge-amber",
        "Good": "badge-green",
        "Average": "badge-amber",
        "Bad": "badge-red",
        "Responded": "badge-green",
        "Failed": "badge-red",
        "Ready": "badge-green",
        "Building": "badge-amber",
        "NotBuilt": "badge-gray",
    }
    return mapping.get(str(value), "badge-blue")
