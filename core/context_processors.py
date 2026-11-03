"""
Template context processors.
Exposes active nav key and simple user-role helpers to every template.
"""


def nav(request):
    """Return the active page/app slug for sidebar highlighting."""
    app_name = ""
    try:
        match = request.resolver_match
        if match is not None:
            app_name = match.app_name or match.namespace or match.url_name or ""
    except Exception:
        app_name = ""

    user = getattr(request, "user", None)
    role_name = ""
    if user and user.is_authenticated:
        try:
            role_name = user.role.name if user.role_id else ""
        except Exception:
            role_name = ""
    return {
        "active_nav": app_name,
        "current_role": role_name,
    }
