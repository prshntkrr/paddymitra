from django.urls import path

from . import views

app_name = "documents"

urlpatterns = [
    path("", views.DocumentListView.as_view(), name="list"),
    path("new/", views.DocumentCreateView.as_view(), name="create"),
    path("review/", views.ReviewQueueView.as_view(), name="review_queue"),
    path("<int:pk>/", views.DocumentDetailView.as_view(), name="detail"),
    path("<int:pk>/edit/", views.DocumentUpdateView.as_view(), name="edit"),
    path("<int:pk>/delete/", views.DocumentDeleteView.as_view(), name="delete"),
    path("<int:pk>/publish/", views.publish_document, name="publish"),
    path("<int:pk>/unpublish/", views.unpublish_document, name="unpublish"),
    path("<int:pk>/approve/", views.approve_document, name="approve"),
    path("<int:pk>/reject/", views.reject_document, name="reject"),
    path("<int:pk>/submit-review/", views.submit_for_review, name="submit_review"),
]
