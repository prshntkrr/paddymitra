from django.urls import path

from . import views

app_name = "hooks"

urlpatterns = [
    path("", views.HookListView.as_view(), name="list"),
    path("new/", views.HookCreateView.as_view(), name="create"),
    path("<int:pk>/", views.HookDetailView.as_view(), name="detail"),
    path("<int:pk>/edit/", views.HookUpdateView.as_view(), name="edit"),
    path("<int:pk>/delete/", views.HookDeleteView.as_view(), name="delete"),
    path("<int:pk>/toggle/", views.toggle_hook, name="toggle"),
    # The canonical inbound URL is /api/v1/whatsapp/pickyassist/ (see
    # irri_ai/urls.py). This one stays routed because it is what any gateway
    # configured before today posts to, and a webhook that starts 404ing is a
    # farmer whose messages vanish.
    path("webhook/pickyassist/", views.PickyAssistWebhookView.as_view(), name="webhook_pickyassist"),
]
