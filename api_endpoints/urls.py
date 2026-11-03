from django.urls import path

from . import views
# The WhatsApp integration is an API, and this is where someone looks for
# one. Its views live in hooks/ with the model and the service they use.
from hooks import console as whatsapp_console

app_name = "api_endpoints"

urlpatterns = [
    path("", views.APIListView.as_view(), name="list"),
    path("whatsapp/", whatsapp_console.whatsapp_api, name="whatsapp"),
    path("whatsapp/keys/new/", whatsapp_console.create_whatsapp_key,
         name="whatsapp_key_create"),
    path("whatsapp/keys/<int:pk>/rotate/", whatsapp_console.rotate_whatsapp_key,
         name="whatsapp_key_rotate"),
    path("whatsapp/keys/<int:pk>/revoke/", whatsapp_console.revoke_whatsapp_key,
         name="whatsapp_key_revoke"),
    path("create/", views.create_api_from_query, name="create_from_query"),
    path("<uuid:public_id>/", views.APIDetailView.as_view(), name="detail"),
    path("<uuid:public_id>/delete/", views.delete_api, name="delete"),
    path("<uuid:public_id>/rotate/", views.rotate_token, name="rotate"),
]
