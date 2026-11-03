from django.contrib import messages
from django.contrib.auth import views as auth_views
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import CreateView, DeleteView, ListView, UpdateView

from core.mixins import RoleRequiredMixin

from .forms import EmailAuthenticationForm, UserCreateForm, UserUpdateForm
from .models import User


class LoginView(auth_views.LoginView):
    template_name = "registration/login.html"
    authentication_form = EmailAuthenticationForm
    redirect_authenticated_user = True


class LogoutView(auth_views.LogoutView):
    next_page = "accounts:login"


class AdminOnlyMixin(RoleRequiredMixin):
    required_roles = ("Administrator",)


class UserListView(AdminOnlyMixin, ListView):
    model = User
    template_name = "accounts/users_list.html"
    context_object_name = "users"
    paginate_by = 50

    def get_queryset(self):
        return (
            User.objects.select_related("role")
            .exclude(is_superuser=False, role__isnull=True, email="")
            .order_by("-is_active", "email")
        )

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        qs = User.objects.all()
        ctx["total_users"] = qs.count()
        ctx["active_users"] = qs.filter(is_active=True, status="Active").count()
        ctx["inactive_users"] = qs.filter(status="Inactive").count()
        from .models import Role
        ctx["roles_defined"] = Role.objects.count()
        ctx["all_roles"] = Role.objects.all()
        return ctx


class UserCreateView(AdminOnlyMixin, CreateView):
    model = User
    form_class = UserCreateForm
    template_name = "accounts/user_form.html"
    success_url = reverse_lazy("accounts:users_list")

    def form_valid(self, form):
        messages.success(self.request, "User created.")
        return super().form_valid(form)


class UserUpdateView(AdminOnlyMixin, UpdateView):
    model = User
    form_class = UserUpdateForm
    template_name = "accounts/user_form.html"
    success_url = reverse_lazy("accounts:users_list")

    def form_valid(self, form):
        messages.success(self.request, "User updated.")
        return super().form_valid(form)


class UserDeleteView(AdminOnlyMixin, DeleteView):
    model = User
    template_name = "accounts/user_confirm_delete.html"
    success_url = reverse_lazy("accounts:users_list")

    def form_valid(self, form):
        messages.success(self.request, "User deleted.")
        return super().form_valid(form)
