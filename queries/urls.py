from django.urls import path

from . import views

app_name = "queries"

urlpatterns = [
    path("", views.QueryListView.as_view(), name="list"),
    path("export/", views.export_queries, name="export"),
    # The testing playground. Nothing it runs is saved — see
    # services/query_test.py.
    path("test/", views.test_query_page, name="test_query"),
    path("test/run/", views.test_query_run, name="test_query_run"),
    path("demo/", views.QueryDemoView.as_view(), name="demo"),
    path("demo/chat/", views.demo_chat, name="demo_chat"),
    path("demo/chat/<int:pk>/feedback/", views.set_chat_feedback, name="set_chat_feedback"),
    path("chat/", views.QueryChatView.as_view(), name="chat"),
    path("glossary/", views.glossary_manage, name="glossary"),
    path("synonyms/", views.synonyms_manage, name="synonyms"),
    path("faq/", views.faq_manage, name="faq"),
    path("faq/reembed/", views.faq_reembed, name="faq_reembed"),
    path("glossary/download/", views.glossary_download, name="glossary_download"),
    path("translation-test/", views.translation_test, name="translation_test"),
    path("batch-test/", views.batch_test_page, name="batch_test"),
    path("batch-test/run/", views.batch_test_run, name="batch_test_run"),
    path("<int:pk>/", views.QueryDetailView.as_view(), name="detail"),
    path("<int:pk>/flag/", views.flag_query, name="flag"),
    path("<int:pk>/feedback/", views.set_feedback, name="set_feedback"),
]
