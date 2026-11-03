"""The Query Listing, downloaded.

The thing worth testing is not that a file comes back — it is that the file
holds the SAME queries the screen was showing. A download that quietly ignores
the filter, or quietly stops at the end of page one, looks like a complete
record and is not.
"""
import io

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from openpyxl import load_workbook

from queries.models import Query
from queries.services.query_export import COLUMNS

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class QueryExportTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user("tester", password="x")
        # More than one page's worth (paginate_by = 25), across two categories.
        for i in range(30):
            Query.objects.create(
                district="Cuttack" if i % 2 else "Puri",
                category="Rice Variety" if i % 2 else "Disease",
                original_query_text=f"ପ୍ରଶ୍ନ {i}",
                translated_query_text=f"Which variety is suitable for upland? {i}",
                ai_response_text=f"CR Dhan 100 is suitable for upland. {i}",
                status="Responded",
                pipeline_trace={
                    "canonical_query": f"rice variety recommended for upland {i}",
                    "stt_backend": "indic-conformer",
                    "faq": {"id": 7, "score": 0.94, "question": "which variety for upland"},
                    "retrieved_chunks": [{"document_title": "Varietal sheet"}],
                },
            )

    def setUp(self):
        self.client.force_login(self.user)

    def sheet_for(self, **params):
        response = self.client.get(reverse("queries:export"), params)
        self.assertEqual(200, response.status_code)
        self.assertEqual(XLSX, response["Content-Type"])
        self.assertIn("attachment; filename=", response["Content-Disposition"])
        book = load_workbook(io.BytesIO(response.content))
        return book["Queries"]

    def test_the_file_opens_and_has_the_headers(self):
        sheet = self.sheet_for()
        headers = [c.value for c in next(sheet.iter_rows(max_row=1))]

        self.assertEqual([title for title, _, _ in COLUMNS], headers)
        self.assertEqual("A2", sheet.freeze_panes)
        self.assertTrue(sheet.column_dimensions["A"].width)

    def test_every_query_is_exported_not_just_the_first_page(self):
        sheet = self.sheet_for()

        self.assertEqual(Query.objects.count() + 1, sheet.max_row)
        self.assertEqual(30, Query.objects.count())

    def test_a_filter_narrows_the_file_the_same_way_it_narrows_the_page(self):
        sheet = self.sheet_for(category="Disease")
        listing = self.client.get(reverse("queries:list"), {"category": "Disease"})

        self.assertEqual(15, sheet.max_row - 1)
        self.assertEqual(15, listing.context["paginator"].count)

    def test_a_search_term_narrows_it_too(self):
        sheet = self.sheet_for(q="upland? 7")

        self.assertEqual(1, sheet.max_row - 1)

    def test_the_filters_combine_exactly_as_the_listing_combines_them(self):
        params = {"category": "Rice Variety", "district": "Cuttack"}
        sheet = self.sheet_for(**params)
        listing = self.client.get(reverse("queries:list"), params)

        self.assertEqual(listing.context["paginator"].count, sheet.max_row - 1)

    def test_a_filter_matching_nothing_gives_a_file_with_only_headers(self):
        sheet = self.sheet_for(district="Nowhere")

        self.assertEqual(1, sheet.max_row)

    def test_the_useful_fields_are_actually_in_the_row(self):
        sheet = self.sheet_for(q="upland? 3")
        headers = [c.value for c in next(sheet.iter_rows(max_row=1))]
        row = dict(zip(headers, [c.value for c in next(sheet.iter_rows(min_row=2, max_row=2))]))

        self.assertTrue(row["Query ID"].startswith("Q-"))
        self.assertIn("Which variety is suitable for upland?", row["Query (English)"])
        self.assertIn("ପ୍ରଶ୍ନ", row["Original Query"])
        self.assertIn("CR Dhan 100", row["AI Response"])
        self.assertIn("rice variety recommended for upland", row["Canonical Rewrite"])
        self.assertEqual("indic-conformer", row["STT Backend"])
        self.assertIn("FAQ-7", row["FAQ Match"])
        self.assertEqual("Varietal sheet", row["Sources / Documents"])
        self.assertIsNotNone(row["Date / Time"])

    def test_the_export_is_one_bounded_set_of_queries(self):
        # Session, user, the rows, one prefetch for the source documents — and
        # that is all, however many rows there are. The m2m is prefetched and
        # everything else is read off the row already in hand, so a query per
        # row cannot creep back in.
        with self.assertNumQueries(4):
            self.client.get(reverse("queries:export"))

        keep = list(Query.objects.values_list("pk", flat=True)[:5])
        Query.objects.exclude(pk__in=keep).delete()
        with self.assertNumQueries(4):
            self.client.get(reverse("queries:export"))

    def test_control_characters_do_not_break_the_file(self):
        Query.objects.create(translated_query_text="bell \x07 and \x0c form feed",
                             ai_response_text="fine")
        sheet = self.sheet_for(q="bell")

        self.assertEqual(1, sheet.max_row - 1)

    def test_the_listing_offers_the_download_with_its_filters_attached(self):
        response = self.client.get(reverse("queries:list"), {"category": "Disease"})
        html = response.content.decode()

        self.assertIn("Download Excel", html)
        self.assertIn(f'{reverse("queries:export")}?category=Disease', html)

    def test_signing_in_is_required(self):
        self.client.logout()
        response = self.client.get(reverse("queries:export"))

        self.assertEqual(302, response.status_code)
