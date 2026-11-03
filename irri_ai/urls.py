from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from api_endpoints.views import dispatch_api

from hooks.api import whatsapp_message
from hooks.views import PickyAssistWebhookView, whatsapp_health

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include(("accounts.urls", "accounts"), namespace="accounts")),
    path("documents/", include(("documents.urls", "documents"), namespace="documents")),
    path("keywords/", include(("keywords.urls", "keywords"), namespace="keywords")),
    path("catalogues/", include(("catalogues.urls", "catalogues"), namespace="catalogues")),
    path("hooks/", include(("hooks.urls", "hooks"), namespace="hooks")),
    path("queries/", include(("queries.urls", "queries"), namespace="queries")),
    path("image-bank/", include(("image_bank.urls", "image_bank"), namespace="image_bank")),
    path("apis/", include(("api_endpoints.urls", "api_endpoints"), namespace="api_endpoints")),
    # Public dispatch endpoint (token auth, no login)
    path("api/v1/q/<uuid:public_id>/", dispatch_api, name="api_dispatch"),
    # other /api/v1/ routes because that is what an integrator looks for, and
    # kept out of the hooks/ namespace so the public surface is one place.
    path("api/v1/whatsapp/pickyassist/", PickyAssistWebhookView.as_view(),
         name="whatsapp_inbound"),
    # The provider-independent endpoint: any gateway or service that can POST
    # JSON, authenticated by a generic API key, no hook involved.
    path("api/v1/whatsapp/message/", whatsapp_message, name="whatsapp_message"),
    path("api/v1/whatsapp/health/", whatsapp_health, name="whatsapp_health"),
    path("", include(("dashboard.urls", "dashboard"), namespace="dashboard")),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
