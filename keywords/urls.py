from django.urls import path

from . import views

app_name = "keywords"

urlpatterns = [
    path("", views.KeywordListView.as_view(), name="list"),
    path("new/", views.KeywordCreateView.as_view(), name="create"),
    path("<int:pk>/", views.KeywordDetailView.as_view(), name="detail"),
    path("<int:pk>/edit/", views.KeywordUpdateView.as_view(), name="edit"),
    path("<int:pk>/delete/", views.KeywordDeleteView.as_view(), name="delete"),
]
