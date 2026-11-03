from django.urls import path

from . import views

app_name = "catalogues"

urlpatterns = [
    path("", views.CatalogueListView.as_view(), name="list"),
    path("beta-test/", views.beta_test, name="beta_test"),
    path("transcribe/", views.transcribe_audio, name="transcribe"),
    path("new/", views.CatalogueCreateView.as_view(), name="create"),
    path("<slug:slug>/", views.CatalogueDetailView.as_view(), name="detail"),
    path("<slug:slug>/edit/", views.CatalogueUpdateView.as_view(), name="edit"),
    path("<slug:slug>/delete/", views.CatalogueDeleteView.as_view(), name="delete"),
    path("<slug:slug>/build-vdb/", views.build_vdb, name="build_vdb"),
    path("<slug:slug>/search/", views.CatalogueSearchView.as_view(), name="search"),
]
