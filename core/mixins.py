"""
Role-based access control mixins.
"""
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied


class RoleRequiredMixin(LoginRequiredMixin):
    """
    CBV mixin: only lets users whose role name is in `required_roles` through.
    Administrators always pass.
    """

    required_roles = ()

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return self.handle_no_permission()
        role = getattr(request.user, "role", None)
        role_name = role.name if role else ""
        if role_name == "Administrator":
            return super().dispatch(request, *args, **kwargs)
        if self.required_roles and role_name not in self.required_roles:
            raise PermissionDenied(
                f"Your role ({role_name or 'None'}) is not permitted to access this page."
            )
        return super().dispatch(request, *args, **kwargs)
