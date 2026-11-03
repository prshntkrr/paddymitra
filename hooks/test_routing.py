"""Automatic catalogue routing for the generic WhatsApp endpoint.

The farmer names no corpus. The question has to say which one it is, and when it
does not say clearly, the default answers — guessing between two corpora is how
a farmer gets an answer from the wrong one.
"""
from django.test import SimpleTestCase, TestCase, override_settings

from catalogues.models import Catalogue
from hooks import routing

ROUTES = "nutrient:5,variety:6,pest:7"


class ClassifyTests(SimpleTestCase):
    """Which category a question belongs to, before any catalogue exists."""

    def category(self, question):
        return routing.classify(question)["category"]

    # ── the four required cases ───────────────────────────────────────────
    def test_a_dose_question_is_nutrient(self):
        self.assertEqual("nutrient",
                         self.category("How much urea should I apply at tillering?"))

    def test_a_variety_choice_question_is_variety(self):
        self.assertEqual("variety",
                         self.category("Which rice variety is suitable for medium land?"))

    def test_a_disease_cause_question_is_pest(self):
        self.assertEqual("pest", self.category("What causes leaf blast?"))

    def test_a_general_agronomy_question_matches_nothing(self):
        # Spacing belongs to none of the three corpora, so it must not be
        # forced into one.
        self.assertEqual("", self.category(
            "What spacing should I keep for transplanting rice?"))

    # ── why this is not keyword matching ──────────────────────────────────
    def test_a_variety_question_that_names_a_disease_is_still_variety(self):
        # "blast" is pest vocabulary, but what is being ASKED for is a variety.
        self.assertEqual("variety", self.category("Which variety resists blast?"))

    def test_a_dose_question_that_names_a_variety_is_still_nutrient(self):
        self.assertEqual("nutrient",
                         self.category("How much fertilizer for BINA Dhan variety?"))

    def test_a_control_question_is_pest_even_with_no_pest_named(self):
        self.assertEqual("pest",
                         self.category("How do I control the insects attacking my crop?"))

    def test_an_evenly_split_question_routes_nowhere(self):
        # Subject in two categories, decisive intent in neither: the honest
        # answer is the default, not a coin toss.
        self.assertEqual("", self.category("Which fertilizer variety is best?"))

    def test_an_empty_question_routes_nowhere(self):
        self.assertEqual("", self.category(""))

    def test_odia_routes_on_its_subject_vocabulary(self):
        # Intent phrases are English-only, so an Odia question routes on the
        # glossary's own spelling of the subject. Without this, every real
        # message would go to the default.
        #
        # The glossary is DATA, not code: odia_glossary.json is generated per
        # deployment and is absent from some checkouts. Where it is missing there
        # are no Odia surfaces to match, and this states that plainly instead of
        # failing as though the router were broken.
        from queries.services.odia_glossary import get_variants
        if not get_variants("urea"):
            self.skipTest("odia_glossary.json is not installed in this checkout, "
                          "so no Odia spellings exist to match")
        self.assertEqual("nutrient", self.category("ଧାନରେ ୟୁରିଆ କେତେ ଦେବି?"))


class RoutesConfigTests(SimpleTestCase):
    """WHATSAPP_CATALOGUE_ROUTES is the one place routing is configured."""

    @override_settings(WHATSAPP_CATALOGUE_ROUTES=ROUTES)
    def test_the_configured_pairs_are_read(self):
        self.assertEqual({"nutrient": 5, "variety": 6, "pest": 7}, routing.routes())

    @override_settings(WHATSAPP_CATALOGUE_ROUTES="")
    def test_unset_means_no_routing(self):
        self.assertEqual({}, routing.routes())

    @override_settings(WHATSAPP_CATALOGUE_ROUTES="nutrient:5,variety,pest:seven")
    def test_a_malformed_pair_is_skipped_not_fatal(self):
        # A typo in one route must not stop the others answering.
        self.assertEqual({"nutrient": 5}, routing.routes())


@override_settings(WHATSAPP_CATALOGUE_ROUTES=ROUTES)
class RouteTests(TestCase):
    """The catalogue a question actually reaches."""

    def setUp(self):
        self.nutrient = Catalogue.objects.create(pk=5, name="Nutrient Management")
        self.variety = Catalogue.objects.create(pk=6, name="Varietal recommendation")
        self.pest = Catalogue.objects.create(pk=7, name="Insect-pest and disease")
        self.default = Catalogue.objects.create(pk=4, name="Rice Knowledge Base")

    def route(self, question):
        catalogue, category = routing.route(question, default=self.default)
        return catalogue.pk, category

    def test_nutrient_question_reaches_catalogue_5(self):
        self.assertEqual((5, "nutrient"),
                         self.route("How much urea should I apply at tillering?"))

    def test_variety_question_reaches_catalogue_6(self):
        self.assertEqual((6, "variety"),
                         self.route("Which rice variety is suitable for medium land?"))

    def test_pest_question_reaches_catalogue_7(self):
        self.assertEqual((7, "pest"), self.route("What causes leaf blast?"))

    def test_a_general_question_falls_back_to_the_default(self):
        self.assertEqual((4, ""), self.route(
            "What spacing should I keep for transplanting rice?"))

    def test_a_routed_catalogue_that_does_not_exist_falls_back(self):
        # Configuration can outlive a catalogue. Answering from the default is
        # better than raising at the farmer.
        self.pest.delete()
        self.assertEqual((4, ""), self.route("What causes leaf blast?"))

    @override_settings(WHATSAPP_CATALOGUE_ROUTES="")
    def test_with_no_routes_configured_everything_uses_the_default(self):
        self.assertEqual((4, ""),
                         self.route("How much urea should I apply at tillering?"))

    @override_settings(WHATSAPP_CATALOGUE_ROUTES="nutrient:5")
    def test_only_configured_categories_are_selectable(self):
        # pest is recognised by the classifier but has no catalogue, so the
        # default answers rather than a corpus nobody chose.
        self.assertEqual((4, ""), self.route("What causes leaf blast?"))
        self.assertEqual((5, "nutrient"),
                         self.route("How much urea should I apply at tillering?"))


@override_settings(WHATSAPP_CATALOGUE_ROUTES=ROUTES)
class PlanTests(TestCase):
    """One corpus normally; two when the question genuinely spans two."""

    def setUp(self):
        Catalogue.objects.create(pk=5, name="Nutrient Management")
        Catalogue.objects.create(pk=6, name="Varietal recommendation")
        Catalogue.objects.create(pk=7, name="Insect-pest and disease")
        self.default = Catalogue.objects.create(pk=4, name="Rice Knowledge Base")

    def plan(self, question):
        return routing.plan(question, default=self.default)

    # ── single ────────────────────────────────────────────────────────────
    def test_a_dose_question_uses_one_catalogue(self):
        decision = self.plan("How much urea should I apply at tillering?")

        self.assertEqual("nutrient", decision["primary"])
        self.assertEqual([], decision["secondary"])
        self.assertEqual([5], decision["catalogue_ids"])

    def test_a_variety_question_uses_one_catalogue(self):
        self.assertEqual([6], self.plan(
            "Which variety is suitable for medium land?")["catalogue_ids"])

    def test_a_disease_question_uses_one_catalogue(self):
        self.assertEqual([7], self.plan("What causes leaf blast?")["catalogue_ids"])

    def test_a_variety_named_in_a_dose_question_does_not_add_a_catalogue(self):
        # The variety is a qualifier on a dose question, not a second question.
        decision = self.plan("How much fertilizer for BINA Dhan variety?")

        self.assertEqual("nutrient", decision["primary"])
        self.assertEqual([5], decision["catalogue_ids"])

    # ── two ───────────────────────────────────────────────────────────────
    def test_variety_plus_pest(self):
        decision = self.plan("Which rice variety is resistant to blast?")

        self.assertEqual("variety", decision["primary"])
        self.assertEqual(["pest"], decision["secondary"])
        self.assertEqual([6, 7], decision["catalogue_ids"])

    def test_variety_plus_nutrient(self):
        decision = self.plan(
            "Which variety should I use and what fertilizer should I apply?")

        self.assertEqual({"variety", "nutrient"},
                         {decision["primary"], *decision["secondary"]})
        self.assertEqual({6, 5}, set(decision["catalogue_ids"]))

    def test_three_way_question_uses_three_catalogues(self):
        decision = self.plan(
            "Which variety should I grow, how much urea should I apply, "
            "and how do I control stem borer?")

        self.assertEqual(3, len(decision["catalogue_ids"]))
        self.assertEqual({5, 6, 7}, set(decision["catalogue_ids"]))

    # ── uncertainty and fallback ──────────────────────────────────────────
    def test_an_ambiguous_question_uses_the_default_alone(self):
        # Reaching for several corpora when unsure is guessing wider, not better.
        decision = self.plan("Which fertilizer variety is best?")

        self.assertEqual("", decision["primary"])
        self.assertEqual([4], decision["catalogue_ids"])

    def test_a_general_question_uses_the_default_alone(self):
        self.assertEqual([4], self.plan(
            "What spacing should I keep for transplanting rice?")["catalogue_ids"])

    def test_a_missing_secondary_catalogue_leaves_the_primary_alone(self):
        Catalogue.objects.filter(pk=7).delete()
        decision = self.plan("Which rice variety is resistant to blast?")

        self.assertEqual("variety", decision["primary"])
        self.assertEqual([], decision["secondary"])
        self.assertEqual([6], decision["catalogue_ids"])

    def test_route_still_returns_one_catalogue(self):
        # The single-catalogue façade the hook path uses is unchanged.
        catalogue, category = routing.route(
            "Which rice variety is resistant to blast?", default=self.default)

        self.assertEqual(6, catalogue.pk)
        self.assertEqual("variety", category)


@override_settings(WHATSAPP_CATALOGUE_ROUTES=ROUTES)
class SeedManagementTests(TestCase):
    """What to DO with seed, as opposed to WHICH seed to buy.

    Seed treatment, soaking, germination and seedling emergence live in the
    general corpus, not in the varietal sheets — so the seed-management
    category has no route of its own and the default IS its catalogue. The
    category still exists, because knowing the question is about handling seed
    is what keeps it off the variety corpus.
    """

    def setUp(self):
        Catalogue.objects.create(pk=5, name="Nutrient Management")
        Catalogue.objects.create(pk=6, name="Varietal recommendation")
        Catalogue.objects.create(pk=7, name="Insect-pest and disease")
        self.default = Catalogue.objects.create(pk=4, name="Rice Knowledge Base")

    def ids(self, question):
        return routing.plan(question, default=self.default)["catalogue_ids"]

    def category(self, question):
        return routing.classify(question)["category"]

    # ── seed management goes to the general corpus ────────────────────────
    def test_general_seed_management(self):
        self.assertEqual([4], self.ids("How should I manage rice seeds?"))
        self.assertEqual([4], self.ids(
            "Please give advice on how to properly manage rice seeds."))

    def test_seed_treatment(self):
        self.assertEqual([4], self.ids(
            "How should rice seeds be treated before sowing?"))

    def test_seed_preparation(self):
        self.assertEqual([4], self.ids(
            "How should I prepare rice seed before sowing?"))
        self.assertEqual([4], self.ids("What should I do to seeds before sowing?"))

    def test_germination_and_emergence(self):
        self.assertEqual([4], self.ids("How can I improve seed germination?"))
        self.assertEqual([4], self.ids("How can I ensure uniform seedling emergence?"))

    def test_these_are_recognised_as_seed_management_not_as_nothing(self):
        # The catalogue is the default either way, but the category is what a
        # log reader — and any future route — needs to see.
        self.assertEqual("seed_management",
                         self.category("How should rice seeds be treated before sowing?"))

    def test_a_configured_route_for_seed_management_is_honoured(self):
        with override_settings(
                WHATSAPP_CATALOGUE_ROUTES="nutrient:5,variety:6,pest:7,seed_management:5"):
            self.assertEqual([5], self.ids("How should I manage rice seeds?"))

    # ── the word "seed" does not decide anything on its own ───────────────
    def test_choosing_a_variety_is_still_variety(self):
        self.assertEqual([6], self.ids("Which variety should I plant?"))
        self.assertEqual([6], self.ids("Which rice variety is suitable for drought?"))

    def test_which_seed_resists_a_disease_is_variety_plus_pest(self):
        # "seed" here is the thing being CHOSEN, and the disease qualifies the
        # choice — both corpora are needed.
        decision = routing.plan("Which seed is resistant to blast?",
                                default=self.default)

        self.assertEqual("variety", decision["primary"])
        self.assertEqual(["pest"], decision["secondary"])
        self.assertEqual([6, 7], decision["catalogue_ids"])

    def test_a_seed_rate_question_is_not_a_variety_question(self):
        # Its evidence is agronomy, not the varietal sheets.
        self.assertNotEqual([6], self.ids("What is the seed rate for DSR?"))
        self.assertEqual([4], self.ids("What is the seed rate for DSR?"))

    def test_an_ambiguous_seed_question_does_not_become_variety(self):
        self.assertNotEqual([6], self.ids("Tell me about rice seed."))

    # ── nothing else moved ────────────────────────────────────────────────
    def test_nutrient_routing_is_unchanged(self):
        self.assertEqual([5], self.ids("How much urea should I apply at tillering?"))

    def test_pest_routing_is_unchanged(self):
        self.assertEqual([7], self.ids("What causes leaf blast?"))

    def test_the_ambiguous_case_is_unchanged(self):
        self.assertEqual([4], self.ids("Which fertilizer variety is best?"))


@override_settings(WHATSAPP_CATALOGUE_ROUTES=ROUTES)
class SeedVersusVarietyTests(TestCase):
    """The same word, two questions, two corpora.

    "Seed" is what a farmer calls the thing they buy AND the thing they treat
    before sowing. What separates them is never the noun — it is whether the
    question is about CHOOSING one (which land, which stress, which yield) or
    about HANDLING one (treat, soak, store, germinate).
    """

    def setUp(self):
        Catalogue.objects.create(pk=5, name="Nutrient Management")
        Catalogue.objects.create(pk=6, name="Varietal recommendation")
        Catalogue.objects.create(pk=7, name="Insect-pest and disease")
        self.default = Catalogue.objects.create(pk=4, name="Rice Knowledge Base")

    def ids(self, question):
        return routing.plan(question, default=self.default)["catalogue_ids"]

    # ── choosing a seed is a variety question ─────────────────────────────
    def test_what_seed_for_a_land_type(self):
        self.assertEqual([6], self.ids("What seed should I use for high land?"))
        self.assertEqual([6], self.ids("What seed is good for lowland?"))

    def test_seed_for_an_ecology(self):
        self.assertEqual([6], self.ids("Which seed is good for rainfed land?"))
        self.assertEqual([6], self.ids("Which variety is suitable for upland?"))

    def test_seed_chosen_for_yield(self):
        self.assertEqual([6], self.ids(
            "What type of seed gives good yield in upland?"))

    def test_seed_chosen_for_a_stress_or_disease(self):
        self.assertEqual([6], self.ids("Which seed is drought tolerant?"))
        # a disease characteristic needs the pest corpus alongside the varietal one
        self.assertEqual([6, 7], self.ids("Which seed is resistant to blast?"))

    # ── handling a seed is not ────────────────────────────────────────────
    def test_treatment_preparation_and_storage_stay_general(self):
        for question in ("How should I treat seed before sowing?",
                         "How should I prepare rice seed before sowing?",
                         "How should I store seed after harvest?",
                         "How do I check seed quality?",
                         "How should I manage rice seeds?"):
            with self.subTest(question=question):
                self.assertEqual([4], self.ids(question))

    def test_a_seed_rate_question_is_still_not_variety(self):
        # "What IS THE seed rate" — the words between "what" and "seed" are
        # what keep this off the variety corpus.
        self.assertEqual([4], self.ids("What is the seed rate for DSR?"))

    # ── the land words must not swallow other questions ───────────────────
    def test_a_flood_recovery_question_is_not_a_variety_question(self):
        # "flood tolerant" is a variety trait; a flooded field is a disaster.
        self.assertNotEqual([6], self.ids(
            "What should I do after my field is flooded?"))

    def test_a_dose_question_naming_a_land_type_is_still_nutrient(self):
        self.assertEqual([5], self.ids("How much urea should I apply in lowland?"))
