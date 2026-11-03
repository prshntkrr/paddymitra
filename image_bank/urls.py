from django.urls import path

from . import views

app_name = "image_bank"

urlpatterns = [
    path("", views.CataloguePickerView.as_view(), name="list"),
    path("purge-all/", views.purge_all_vectors, name="purge_all"),
    path("<slug:slug>/", views.GroupListView.as_view(), name="groups"),
    path("<slug:slug>/upload/", views.GroupCreateView.as_view(), name="group_create"),
    path("<slug:slug>/reindex/", views.reindex_catalogue, name="reindex"),
    path("<slug:slug>/groups/<int:gid>/", views.GroupDetailView.as_view(), name="group_detail"),
    path("<slug:slug>/groups/<int:gid>/add-images/", views.GroupAddImagesView.as_view(), name="group_add_images"),
    path("<slug:slug>/groups/<int:gid>/edit/", views.GroupUpdateDescriptionView.as_view(), name="group_edit"),
    path("<slug:slug>/groups/<int:gid>/delete/", views.GroupDeleteView.as_view(), name="group_delete"),
    path("<slug:slug>/groups/<int:gid>/images/<int:ki_id>/delete/", views.ImageDeleteView.as_view(), name="image_delete"),
]
