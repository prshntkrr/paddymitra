import io
import logging
import time
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.views.generic import DetailView, FormView, ListView, TemplateView

from .forms import QueryDemoForm
from .models import Query

log = logging.getLogger("irri")

_AUDIO_SUFFIXES = (".webm", ".ogg", ".m4a", ".mp3", ".wav")


def _store_recording(query, upload) -> bool:
    """Keep the farmer's own recording against the query they asked with it.

    Called AFTER the pipeline, so the file is named for the query it belongs to
    and only a question that actually ran leaves a recording behind.

    Never raises. The recording is a record of what was asked; the answer is the
    product, and losing the first must not cost the second.
    """
    if not upload or not query:
        return False
    import os
    suffix = os.path.splitext(getattr(upload, "name", "") or "")[1].lower()
    if suffix not in _AUDIO_SUFFIXES:
        suffix = ".webm"
    try:
        upload.seek(0)          # transcription may already have read it to EOF
        query.original_query_voice_file.save(f"q{query.pk}{suffix}", upload, save=True)
        return True
    except Exception as exc:
        log.warning("could not store the farmer's recording for query %s: %s",
                    getattr(query, "pk", "?"), exc)
        return False


# ── The Query Listing queryset ────────────────────────────────────────────
#
# ONE definition of "which queries the user is looking at", because two things
# now read it: the paginated table and the Excel export. Had the export built
# its own filter clause, the file would have drifted from the screen the moment
# either changed — and an export that quietly disagrees with the page it was
# downloaded from is worse than no export.
def filtered_queries(params):
    """Queries matching the listing filters in `params` (a QueryDict/dict)."""
    qs = Query.objects.select_related("hook", "catalogue")
    get = params.get
    if get("category"):
        qs = qs.filter(category=get("category"))
    if get("feedback"):
        qs = qs.filter(farmer_feedback=get("feedback"))
    if get("season"):
        qs = qs.filter(season=get("season"))
    if get("crop_stage"):
        qs = qs.filter(crop_stage=get("crop_stage"))
    if get("district"):
        qs = qs.filter(district__icontains=get("district"))
    if get("q"):
        qs = qs.filter(translated_query_text__icontains=get("q"))
    return qs.order_by("-timestamp")


class QueryListView(LoginRequiredMixin, ListView):
    model = Query
    template_name = "queries/list.html"
    context_object_name = "queries"
    paginate_by = 25

    def get_queryset(self):
        return filtered_queries(self.request.GET)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        qs = Query.objects.all()
        ctx["total"] = qs.count()
        ctx["responded"] = qs.filter(status="Responded").count()
        ctx["pending"] = qs.filter(status__in=["Received", "Processing", "Failed"]).count()
        ctx["flagged"] = qs.filter(flagged_for_review=True).count()
        ctx["category_choices"] = Query.CATEGORY_CHOICES
        ctx["feedback_choices"] = Query.FEEDBACK_CHOICES
        ctx["season_choices"] = Query.SEASON_CHOICES
        ctx["crop_stage_choices"] = Query.CROP_STAGE_CHOICES
        return ctx


class QueryDetailView(LoginRequiredMixin, DetailView):
    model = Query
    template_name = "queries/detail.html"
    context_object_name = "qobj"


class QueryDemoView(LoginRequiredMixin, FormView):
    template_name = "queries/demo_form.html"
    form_class = QueryDemoForm

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["recent_queries"] = Query.objects.order_by("-timestamp")[:12]
        ctx["feedback_choices"] = Query.FEEDBACK_CHOICES
        return ctx

    def form_valid(self, form):
        from .services.pipeline import run_pipeline_for_image, run_pipeline_for_text

        image = form.cleaned_data.get("image")
        try:
            if image:
                query = run_pipeline_for_image(
                    catalogue=form.cleaned_data["catalogue"],
                    farmer_language=form.cleaned_data["farmer_language"],
                    uploaded_image=image,
                    district=form.cleaned_data["district"] or "",
                    generate_tts=form.cleaned_data["generate_tts"],
                    user_type=form.cleaned_data["user_type"],
                    user_query_text=form.cleaned_data.get("query_text") or "",
                )
            else:
                query = run_pipeline_for_text(
                    catalogue=form.cleaned_data["catalogue"],
                    farmer_language=form.cleaned_data["farmer_language"],
                    query_text=form.cleaned_data["query_text"],
                    district=form.cleaned_data["district"] or "",
                    generate_tts=form.cleaned_data["generate_tts"],
                    user_type=form.cleaned_data["user_type"],
                )
        except Exception as exc:
            log.exception("Demo pipeline failed")
            messages.error(self.request, f"Pipeline error: {exc}")
            return self.form_invalid(form)

        _store_recording(query, form.cleaned_data.get("query_audio"))
        return redirect("queries:detail", pk=query.pk)


class QueryChatView(LoginRequiredMixin, TemplateView):
    template_name = "queries/chat.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["form"] = QueryDemoForm()
        return ctx

        image = form.cleaned_data.get("image")
        try:
            if image:
                query = run_pipeline_for_image(
                    catalogue=form.cleaned_data["catalogue"],
                    farmer_language=form.cleaned_data["farmer_language"],
                    uploaded_image=image,
                    district=form.cleaned_data["district"] or "",
                    generate_tts=form.cleaned_data["generate_tts"],
                    user_type=form.cleaned_data["user_type"],
                    user_query_text=form.cleaned_data.get("query_text") or "",
                )
            else:
                query = run_pipeline_for_text(
                    catalogue=form.cleaned_data["catalogue"],
                    farmer_language=form.cleaned_data["farmer_language"],
                    query_text=form.cleaned_data["query_text"],
                    district=form.cleaned_data["district"] or "",
                    generate_tts=form.cleaned_data["generate_tts"],
                    user_type=form.cleaned_data["user_type"],
                )
        except Exception as exc:
            log.exception("Demo pipeline failed")
            messages.error(self.request, f"Pipeline error: {exc}")
            return self.form_invalid(form)
        return redirect("queries:detail", pk=query.pk)

@login_required
@require_POST
def demo_chat(request):
    from catalogues.models import Catalogue
    from .services.pipeline import run_pipeline_for_text, stt_transcribe

    catalogue_id = request.POST.get("catalogue")
    if not catalogue_id:
        return JsonResponse({"ok": False, "error": "Catalogue is required."}, status=400)

    catalogue = get_object_or_404(Catalogue, pk=catalogue_id)
    language = request.POST.get("farmer_language", "English")
    user_type = request.POST.get("user_type", "farmer")
    if user_type not in ("farmer", "researcher"):
        user_type = "farmer"
    generate_tts = request.POST.get("generate_tts") in ("on", "true", "1", "True")
    query_text = (request.POST.get("query_text") or "").strip()
    district = request.POST.get("district") or ""

    # The recording may arrive WITH the text (the browser transcribes first, so the
    # farmer can read and correct it before sending) or INSTEAD of it (other
    # clients post the audio raw). Either way the file is kept for storage below —
    # it is only ever uploaded when the farmer actually presses send, so a take
    # they re-recorded because it came out badly never reaches the server.
    audio = request.FILES.get("audio")
    import os
    audio_suffix = os.path.splitext(audio.name)[1].lower() if audio else ""
    if audio_suffix not in _AUDIO_SUFFIXES:
        audio_suffix = ".webm"

    if not query_text and audio:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp:
            for chunk in audio.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        try:
            query_text, _ = stt_transcribe(tmp_path, language_hint=language)
        except Exception as exc:
            return JsonResponse({"ok": False, "error": f"Audio transcription failed: {exc}"}, status=500)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not query_text:
        return JsonResponse({"ok": False, "error": "Enter a question or record audio."}, status=400)

    try:
        query = run_pipeline_for_text(
            catalogue=catalogue,
            farmer_language=language,
            query_text=query_text,
            district=district,
            generate_tts=generate_tts,
            user_type=user_type,
        )
    except Exception as exc:
        log.exception("Demo chat pipeline failed")
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)

    _store_recording(query, audio)

    from .models import ChatQuery
    chat_query = ChatQuery.objects.create(
        query=query,
        language=language,
        question=query_text,
        response=query.ai_response_text_local or query.ai_response_text,
        response_time_seconds=query.response_time_seconds,
    )

    return JsonResponse({
        "ok": True,
        "query_id": query.pk,
        "chat_query_id": chat_query.pk,
        "original_query_text": query.original_query_text,
        "original_query_language": query.original_query_language,
        "original_query_voice_url": (query.original_query_voice_file.url
                                     if query.original_query_voice_file else ""),
        "ai_response_text_local": query.ai_response_text_local,
        "ai_response_text": query.ai_response_text,
        "ai_response_voice_url": query.ai_response_voice_file.url if query.ai_response_voice_file else "",
        "created_at": query.timestamp.isoformat(),
        "feedback": query.farmer_feedback,
    })


@login_required
@require_POST
def set_chat_feedback(request, pk):
    from .models import ChatQuery

    chat_query = get_object_or_404(ChatQuery, pk=pk)
    feedback = request.POST.get("feedback", "")
    feedback_local = (request.POST.get("feedback_local") or feedback).strip()
    valid_feedback = {choice for choice, _ in ChatQuery.FEEDBACK_CHOICES}
    if feedback not in valid_feedback:
        return JsonResponse({"ok": False, "error": "Invalid feedback."}, status=400)

    chat_query.feedback = feedback
    chat_query.feedback_local = feedback_local[:60]
    chat_query.save(update_fields=["feedback", "feedback_local"])

    faq_note = ""
    if chat_query.query_id:
        legacy_feedback = {
            "Correct": "Good",
            "Need Improvement": "Average",
            "Incorrect": "Bad",
        }[feedback]
        chat_query.query.farmer_feedback = legacy_feedback
        chat_query.query.save(update_fields=["farmer_feedback"])
        # "Correct" here is the same judgement as "Good" on the query page, so it
        # feeds the FAQ the same way.
        faq_note = _sync_faq_with_feedback(chat_query.query, legacy_feedback, request.user)

    return JsonResponse({
        "ok": True,
        "feedback": chat_query.feedback,
        "feedback_local": chat_query.feedback_local,
        "faq_note": faq_note.strip(),
    })


@login_required
@require_POST
def flag_query(request, pk):
    role = getattr(request.user, "role", None)
    if not role or role.name not in ("Administrator", "Reviewer"):
        return HttpResponseForbidden("Not permitted.")
    q = get_object_or_404(Query, pk=pk)
    q.flagged_for_review = not q.flagged_for_review
    q.status = "Flagged" if q.flagged_for_review else "Responded"
    q.reviewed_by = request.user
    q.save()
    messages.info(request, "Flag toggled.")
    return redirect("queries:detail", pk=pk)


def _sync_faq_with_feedback(query, feedback, user=None) -> str:
    """Keep the FAQ in step with a feedback change. Returns a message fragment.

    Good promotes the answer into the live FAQ; anything else retires whatever
    that query had previously put there. The retirement half matters as much as
    the promotion: an answer rated Good and later corrected to Bad must stop
    being served to other farmers, and without this it would keep answering.

    Never raises — feedback must save even if the FAQ side fails.
    """
    from .services.faq import demote_for_query, promote_from_query
    try:
        if feedback == "Good":
            faq = promote_from_query(query, user=user)
            return " Added to the FAQ." if faq else ""
        if demote_for_query(query):
            return " Removed from the FAQ."
    except Exception:
        log.exception("FAQ sync failed for query %s", getattr(query, "pk", "?"))
    return ""


@login_required
@require_POST
def set_feedback(request, pk):
    q = get_object_or_404(Query, pk=pk)
    fb = request.POST.get("feedback", "")
    if fb in {"Good", "Average", "Bad"}:
        q.farmer_feedback = fb
        q.save()
        messages.success(request, "Feedback saved." + _sync_faq_with_feedback(q, fb, request.user))
    next_url = request.POST.get("next") or request.GET.get("next")
    if next_url:
        return redirect(next_url)
    return redirect("queries:detail", pk=pk)


# ===== Glossary editor (web UI at /queries/glossary/) =====
@login_required
def glossary_manage(request):
    """View / add / update / delete Odia glossary terms in the browser."""
    from queries.services.odia_glossary import (add_variant, delete_term, delete_variant,
                                                get_all_variants, get_glossary,
                                                is_destructive_key, upsert_term)

    # NOTHING is refused. The glossary is how the operator overrides the
    # translator, so if they decide a word needs a fixed Odia form, it gets one.
    # A structural word still earns a WARNING rather than a block: it is saved and
    # used, they are just told what it will do. See is_destructive_key().
    def _structural_warning(word):
        return (f"Saved — but '{word}' is a structural English word (an article, "
                f"preposition, auxiliary or common verb). It will be substituted "
                f"everywhere it appears in an answer, which can read oddly in the "
                f"middle of a sentence. Delete it here if the answers get worse.")
    if request.method == "POST":
        action = request.POST.get("action", "upsert")
        english = (request.POST.get("english") or "").strip()
        odia = (request.POST.get("odia") or "").strip()
        if action == "delete":
            ok = delete_term(english)
            messages.info(request, f"Deleted '{english}'." if ok else f"'{english}' not found.")
        elif action == "add_variant":
            # An extra word for the same term. The primary is unchanged — this is
            # only so the reverse direction recognises what a farmer might say.
            if not english or not odia:
                messages.error(request, "Both English and the extra Odia word are required.")
            elif __import__("re").search(r"[A-Za-z]", odia):
                messages.error(request, "The Odia word must not contain English letters.")
            elif add_variant(english, odia):
                messages.success(request, f"Added alternative: {english} → {odia}")
                if is_destructive_key(english):
                    messages.warning(request, _structural_warning(english))
            else:
                messages.info(request, f"'{odia}' is already recorded for {english}.")
        elif action == "delete_variant":
            ok = delete_variant(english, odia)
            messages.info(request, f"Removed '{odia}'." if ok else
                          "Could not remove it — a term must keep at least one word.")
        else:
            if not english or not odia:
                messages.error(request, "Both English and Odia are required.")
            elif __import__("re").search(r"[A-Za-z]", odia):
                messages.error(request, "Odia translation must not contain English letters.")
            elif upsert_term(english, odia):
                messages.success(request, f"Saved: {english.lower()} -> {odia}")
                if is_destructive_key(english):
                    messages.warning(request, _structural_warning(english))
            else:
                messages.error(request, f"Could not save '{english}'.")
        return redirect("queries:glossary")
    # (english, primary, [alternatives]) — the primary is what answers use.
    variants = get_all_variants()
    terms = sorted(
        ((en, vs[0], vs[1:]) for en, vs in variants.items() if vs),
        key=lambda t: t[0].lower(),
    )
    return render(request, "queries/glossary.html", {
        "active_nav": "glossary",
        "terms": terms,
        "count": len(terms),
        "extra_count": sum(len(v) - 1 for v in variants.values() if len(v) > 1),
    })


@login_required
def synonyms_manage(request):
    """View / add / delete question-level synonyms in the browser.

    Separate page from the glossary because the shape is different: the glossary
    is one Odia value per English term, while a synonym set is MANY spellings
    pointing at ONE canonical term.
    """
    import sqlite3

    from queries.services.synonyms import (delete_synonym, get_synonyms,
                                           unverified_terms, upsert_synonym)
    if request.method == "POST":
        action = request.POST.get("action", "upsert")
        canonical = (request.POST.get("canonical") or "").strip()
        variant = (request.POST.get("variant") or "").strip()
        if action == "delete":
            ok = delete_synonym(canonical, variant)
            messages.info(request, f"Removed '{variant}'." if ok else f"'{variant}' not found.")
        elif not canonical or not variant:
            messages.error(request, "Both the canonical term and the variant are required.")
        elif len(variant) < 3:
            messages.error(request, "A variant must be at least 3 characters.")
        elif upsert_synonym(canonical, variant):
            messages.success(request, f"Saved: {variant} → {canonical}")
        else:
            messages.info(request, f"'{variant}' is already listed under {canonical}.")
        return redirect("queries:synonyms")

    data = get_synonyms()
    # Warn about canonical terms the indexed documents never use — those anchor a
    # query to nothing, so the synonym silently buys no retrieval at all.
    missing = []
    try:
        db = Path(settings.CHROMA_PERSIST_DIR) / "chroma.sqlite3"
        if db.exists():
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = con.execute("select c0 from embedding_fulltext_search_content").fetchall()
            con.close()
            missing = unverified_terms("\n".join(r[0] or "" for r in rows))
    except Exception as exc:
        log.warning("synonym corpus check failed: %s", exc)

    terms = sorted(((c, sorted(v, key=str.lower)) for c, v in data.items()),
                   key=lambda kv: kv[0].lower())
    return render(request, "queries/synonyms.html", {
        "terms": terms,
        "count": len(terms),
        "variant_count": sum(len(v) for v in data.values()),
        "missing": missing,
    })


# ===== FAQ editor (web UI at /queries/faq/) =====
@login_required
def faq_manage(request):
    """View / add / edit / retire the answers served instead of running RAG.

    An entry here SHORT-CIRCUITS retrieval for any question that matches it, so
    the page is deliberately explicit about that: how many questions each entry
    has already answered, and where it came from.
    """
    from .models import FAQ
    from .services.faq import save_entry

    if request.method == "POST":
        action = request.POST.get("action", "save")
        pk = request.POST.get("pk") or ""

        if action == "delete" and pk:
            deleted, _ = FAQ.objects.filter(pk=pk).delete()
            messages.info(request, "FAQ deleted." if deleted else "FAQ not found.")
            return redirect("queries:faq")

        if action == "toggle" and pk:
            faq = get_object_or_404(FAQ, pk=pk)
            faq.is_active = not faq.is_active
            faq.save(update_fields=["is_active", "updated_at"])
            messages.info(request, ("FAQ is live again." if faq.is_active
                                    else "FAQ retired — it will no longer answer questions."))
            return redirect("queries:faq")

        question_en = (request.POST.get("question_en") or "").strip()
        answer_en = (request.POST.get("answer_en") or "").strip()
        answer_local = (request.POST.get("answer_local") or "").strip()
        if not question_en:
            messages.error(request, "The English question is required — it is what new questions are matched against.")
            return redirect("queries:faq")
        # An English question with an English answer is a complete entry. The Odia
        # wording is an improvement, not a requirement: without it the answer is
        # translated on each use, exactly as a generated answer would be.
        if not (answer_en or answer_local):
            messages.error(request, "An answer is required. English on its own is fine.")
            return redirect("queries:faq")

        faq = get_object_or_404(FAQ, pk=pk) if pk else FAQ(created_by=request.user)
        faq.question_en = question_en
        faq.question_local = (request.POST.get("question_local") or "").strip()
        faq.answer_en = answer_en
        faq.answer_local = answer_local
        # `language` and `catalogue` are deliberately NOT read from the form — the
        # page no longer asks for them. They are left untouched rather than
        # cleared, because an entry promoted from a Query carries both, and
        # editing its wording here must not silently widen its catalogue scope.
        # A hand-made entry keeps the model defaults: Odia, and all catalogues.
        save_entry(faq)          # recomputes the match keys from the new wording
        messages.success(request, f"Saved FAQ-{faq.pk}.")
        return redirect("queries:faq")

    qs = FAQ.objects.select_related("catalogue", "source_query", "created_by")
    search = (request.GET.get("q") or "").strip()
    if search:
        qs = qs.filter(question_en__icontains=search) | qs.filter(question_local__icontains=search)
    faqs = list(qs.order_by("-is_active", "-updated_at"))

    # An entry with no vector can only ever be matched word-for-word, which is
    # almost never. Surface it rather than let it quietly under-perform.
    unembedded = [f for f in faqs if f.is_active and not f.question_embedding]

    return render(request, "queries/faq.html", {
        "faqs": faqs,
        "count": len(faqs),
        "active_count": sum(1 for f in faqs if f.is_active),
        "served": sum(f.hit_count for f in faqs),
        "unembedded": unembedded,
        "search": search,
        "active_nav": "faq",
    })


@login_required
@require_POST
def faq_reembed(request):
    """Recompute every entry's question vector (after an embedding-model change)."""
    from .services.faq import reembed_all
    try:
        n = reembed_all()
        messages.success(request, f"Re-embedded {n} FAQ entr{'y' if n == 1 else 'ies'}.")
    except Exception as exc:
        log.exception("FAQ re-embed failed")
        messages.error(request, f"Re-embed failed: {exc}")
    return redirect("queries:faq")


@login_required
def glossary_download(request):
    """Download the current glossary as an .xlsx."""
    import io
    from django.http import HttpResponse
    from queries.services.odia_glossary import get_glossary
    try:
        import openpyxl
        from openpyxl.styles import Font
    except ImportError:
        messages.error(request, "openpyxl is not installed on the server.")
        return redirect("queries:glossary")
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Glossary"
    ws.append(["English term", "Odia translation"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for en, od in sorted(get_glossary().items(), key=lambda kv: kv[0].lower()):
        ws.append([en, od])
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 40
    ws.freeze_panes = "A2"
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    resp = HttpResponse(
        buf.read(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    resp["Content-Disposition"] = 'attachment; filename="odia_glossary.xlsx"'
    return resp


@login_required
def export_queries(request):
    """The Query Listing as an .xlsx — every matching query, not the page shown.

    The filters come off the SAME query string the table was rendered with and
    go through the same `filtered_queries()`, so "download what I am looking at"
    means exactly that: filtered in, filtered out, in the same order. Pagination
    is deliberately not applied — the point of the button is the rows the page
    could not show.
    """
    from django.http import HttpResponse
    try:
        from .services.query_export import queries_workbook
    except ImportError:
        messages.error(request, "openpyxl is not installed on the server.")
        return redirect("queries:list")

    queryset = filtered_queries(request.GET)
    try:
        buffer = queries_workbook(queryset)
    except Exception as exc:
        log.exception("query export failed")
        messages.error(request, f"Export failed: {exc}")
        return redirect("queries:list")

    now = timezone.localtime() if settings.USE_TZ else datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M")
    response = HttpResponse(
        buffer.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = f'attachment; filename="queries-{stamp}.xlsx"'
    return response


# ===== Query testing playground (web UI at /queries/test/) ===============
#
# The same pipeline the farmer's question goes through — retrieval, intent,
# candidate filtering, generation — with nothing kept. It exists because
# debugging an answer through the chat page means a Query row, a ChatQuery row
# and an audio file for every attempt, and a reviewer looking at the queries
# list cannot tell a farmer's question from the twentieth attempt at a test.
#
# Persistence is prevented by running inside a transaction that is rolled back;
# see services/query_test.py for why that rather than a flag through the
# pipeline. Nothing in the production path changed to make this page work.
@login_required
def test_query_page(request):
    """The testing page itself."""
    from catalogues.models import Catalogue

    return render(request, "queries/test_query.html", {
        "catalogues": Catalogue.objects.order_by("name"),
        "languages": ["English", "Odia", "Hindi"],
    })


@login_required
@require_POST
def test_query_run(request):
    """Answer one test question and return the answer, sources and trace."""
    from catalogues.models import Catalogue

    from .services.query_test import run_test_query

    question = (request.POST.get("query_text") or "").strip()
    if not question:
        return JsonResponse({"ok": False, "error": "Enter a question to test."},
                            status=400)

    catalogue_id = request.POST.get("catalogue") or ""
    catalogue = Catalogue.objects.filter(pk=catalogue_id).first() if catalogue_id \
        else Catalogue.objects.order_by("name").first()
    if catalogue is None:
        return JsonResponse({"ok": False, "error": "No catalogue is available."},
                            status=400)

    user_type = request.POST.get("user_type", "farmer")
    if user_type not in ("farmer", "researcher"):
        user_type = "farmer"

    result = run_test_query(
        catalogue=catalogue,
        question=question,
        language=request.POST.get("farmer_language") or "English",
        user_type=user_type,
        district=request.POST.get("district") or "",
    )
    result["catalogue"] = catalogue.name
    return JsonResponse(result, status=200 if result.get("ok") else 500)


# ===== Translation test bench (web UI at /queries/translation-test/) =====
#
# Fixing a wrong Odia word means editing the glossary and CHECKING it — but
# asking the question again re-runs retrieval and generation, costs the LLM
# call, and comes back with slightly different English, so you cannot tell
# whether the glossary edit is what changed the Odia. This page runs ONLY
# stage 5 of the pipeline (pipeline.translate) on English you already have, so
# the input is held fixed and the glossary is the only thing that moved.
@login_required
def translation_test(request):
    """Re-translate existing English through the pipeline's own translate()."""
    from queries.services.odia_glossary import reload_glossary, relevant_glossary

    if request.method == "POST":
        from queries.services import pipeline
        text = (request.POST.get("text") or "").strip()
        source = (request.POST.get("source") or "English").strip() or "English"
        target = (request.POST.get("target") or "Odia").strip() or "Odia"
        if not text:
            return JsonResponse({"ok": False, "error": "Nothing to translate."}, status=400)

        # The glossary may have been edited in another tab (or on disk) since
        # this process cached it. Always translate against what is saved NOW.
        reload_glossary()
        try:
            t0 = time.monotonic()
            out = pipeline.translate(text, source, target)
            ms = int((time.monotonic() - t0) * 1000)
        except Exception as exc:
            log.exception("translation test failed")
            return JsonResponse({"ok": False, "error": str(exc)}, status=500)

        terms = relevant_glossary(text) if target == "Odia" else {}
        stored = ""
        pk = (request.POST.get("query") or "").strip()
        if pk.isdigit():
            stored = (Query.objects.filter(pk=pk)
                      .values_list("ai_response_text_local", flat=True).first() or "")
        return JsonResponse({
            "ok": True,
            "output": out,
            "ms": ms,
            "terms": [{"en": en, "od": od} for en, od in sorted(terms.items())],
            "stored": stored,
            "changed": bool(stored) and stored.strip() != (out or "").strip(),
        })

    # GET — prefill from a query when we arrived from its detail page.
    qobj = None
    pk = (request.GET.get("query") or "").strip()
    if pk.isdigit():
        qobj = Query.objects.filter(pk=pk).first()
    text = request.GET.get("text") or (qobj.ai_response_text if qobj else "")
    return render(request, "queries/translation_test.html", {
        "qobj": qobj,
        "text": text,
        "stored": qobj.ai_response_text_local if qobj else "",
        "active_nav": "translation_test",
    })


# ===== Batch Excel test (upload XLSX, run pipeline, download results) =====

@login_required
def batch_test_page(request):
    """Upload an Excel sheet of test questions, run each through the pipeline,
    and download results as XLSX. Nothing is saved to the database."""
    from catalogues.models import Catalogue

    return render(request, "queries/batch_test.html", {
        "catalogues": Catalogue.objects.order_by("name"),
        "languages": ["English", "Odia", "Hindi"],
    })


@login_required
@require_POST
def batch_test_run(request):
    """Process uploaded Excel and return results as a downloadable XLSX."""
    import openpyxl
    import traceback
    from catalogues.models import Catalogue
    from .services.query_test import run_test_query

    print("=" * 60)
    print("BATCH TEST: request received")
    print("=" * 60)

    uploaded = request.FILES.get("excel_file")
    if not uploaded:
        messages.error(request, "Please upload an Excel file.")
        return redirect("queries:batch_test")

    catalogue_id = request.POST.get("catalogue") or ""
    catalogue = Catalogue.objects.filter(pk=catalogue_id).first() if catalogue_id \
        else Catalogue.objects.order_by("name").first()
    print(f"BATCH TEST: catalogue={catalogue}")
    if catalogue is None:
        messages.error(request, "No catalogue available.")
        return redirect("queries:batch_test")

    language = request.POST.get("farmer_language") or "English"
    user_type = request.POST.get("user_type", "farmer")
    if user_type not in ("farmer", "researcher"):
        user_type = "farmer"

    # Read the uploaded workbook.
    try:
        wb = openpyxl.load_workbook(uploaded, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception as exc:
        messages.error(request, f"Could not read Excel file: {exc}")
        return redirect("queries:batch_test")

    if len(rows) < 2:
        messages.error(request, "Excel file has no data rows.")
        return redirect("queries:batch_test")

    # Detect question columns from the header row.
    header = [str(c or "").strip().lower() for c in rows[0]]
    odia_col = _find_col(header, ["odia question", "odia", "question_odia"])
    english_col = _find_col(header, ["english question", "english", "question_english", "question"])
    test_id_col = _find_col(header, ["test id", "test_id", "id", "sl", "sl no"])
    category_col = _find_col(header, ["category", "cat"])
    challenge_col = _find_col(header, ["challenge", "challenge / focus", "focus"])

    if odia_col is None and english_col is None:
        messages.error(request, "No 'Odia Question' or 'English Question' column found in the header row.")
        return redirect("queries:batch_test")

    # Build the output workbook.
    out_wb = openpyxl.Workbook()
    out_ws = out_wb.active
    out_ws.title = "Results"

    out_headers = ["Test ID", "Category", "Challenge", "Question", "Language",
                   "Answer (English)", "Answer (Odia)", "Status",
                   "Response Time (s)", "Sources", "Error"]
    out_ws.append(out_headers)

    # Bold header.
    from openpyxl.styles import Font
    bold = Font(bold=True)
    for cell in out_ws[1]:
        cell.font = bold

    total = 0
    for row in rows[1:]:
        # Run both Odia and English questions if present.
        questions = []
        test_id = str(row[test_id_col] or "") if test_id_col is not None else ""
        category = str(row[category_col] or "") if category_col is not None else ""
        challenge = str(row[challenge_col] or "") if challenge_col is not None else ""

        if odia_col is not None and row[odia_col]:
            questions.append((str(row[odia_col]).strip(), "Odia"))
        if english_col is not None and row[english_col]:
            questions.append((str(row[english_col]).strip(), "English"))

        for question, lang in questions:
            if not question:
                continue
            total += 1
            print(f"BATCH TEST [{total}]: {question[:80]} ({lang})")
            result = run_test_query(
                catalogue=catalogue,
                question=question,
                language=lang,
                user_type=user_type,
            )

            sources = ""
            if result.get("ok"):
                src_list = result.get("sources") or []
                sources = "; ".join(
                    f"{s.get('document_title', '')} (chunk {s.get('chunk_index', '')})"
                    for s in src_list[:5]
                )

            out_ws.append([
                test_id,
                category,
                challenge,
                question,
                lang,
                result.get("answer", "") if result.get("ok") else "",
                result.get("answer_local", "") if result.get("ok") else "",
                "OK" if result.get("ok") else "FAIL",
                result.get("response_time_seconds", ""),
                sources,
                result.get("error", "") if not result.get("ok") else "",
            ])

    # Auto-size columns roughly.
    for col in out_ws.columns:
        max_len = max((len(str(cell.value or "")) for cell in col), default=10)
        out_ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)

    # Write to response.
    buf = io.BytesIO()
    out_wb.save(buf)
    buf.seek(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_test_results_{timestamp}.xlsx"

    response = HttpResponse(
        buf.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    log.info("batch test: %d questions processed for catalogue %s", total, catalogue.slug)
    return response


def _find_col(header: list, candidates: list):
    """Find the first column index matching any candidate name."""
    for i, h in enumerate(header):
        if h in candidates:
            return i
    return None
