from collections import Counter, defaultdict
from datetime import timedelta

from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Avg, Count
from django.db.models.functions import TruncMonth
from django.utils import timezone
from django.views.generic import TemplateView

from documents.models import Document
from hooks.models import Hook
from queries.models import Query


CATEGORY_ORDER = ["Rice Variety", "Disease", "Fertiliser", "Water Mgmt", "Others"]
CATEGORY_COLORS = {
    "Rice Variety": "#00833E",
    "Disease": "#DC2626",
    "Fertiliser": "#D97706",
    "Water Mgmt": "#2563EB",
    "Others": "#9CA3AF",
}


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard/dashboard.html"

    # ---------------------------------------------------------------- filters
    def _filtered_queries(self):
        qs = Query.objects.all()
        g = self.request.GET
        if g.get("district"):
            qs = qs.filter(district=g["district"])
        date_from = g.get("from")
        date_to = g.get("to")
        if date_from:
            qs = qs.filter(timestamp__date__gte=date_from)
        if date_to:
            qs = qs.filter(timestamp__date__lte=date_to)
        return qs

    # --------------------------------------------------------------------- main
    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        queries = self._filtered_queries()
        now = timezone.now()

        # ----- KPI cards -----
        docs = Document.objects.all()
        ctx["kpi_docs"] = docs.count()
        ctx["kpi_queries"] = queries.count()
        responded = queries.filter(status="Responded")
        ctx["kpi_responded"] = responded.count()
        ctx["kpi_response_rate"] = round(
            ctx["kpi_responded"] / ctx["kpi_queries"] * 100, 1
        ) if ctx["kpi_queries"] else 0
        fb = queries.exclude(farmer_feedback="")
        ctx["kpi_feedback"] = fb.count()

        fb_counter = Counter(fb.values_list("farmer_feedback", flat=True))
        total_fb = sum(fb_counter.values()) or 1
        ctx["feedback_good_pct"] = round(fb_counter.get("Good", 0) * 100 / total_fb, 1)
        ctx["feedback_avg_pct"] = round(fb_counter.get("Average", 0) * 100 / total_fb, 1)
        ctx["feedback_bad_pct"] = round(fb_counter.get("Bad", 0) * 100 / total_fb, 1)

        # month-over-month deltas (last 30 vs previous 30 days)
        last30 = queries.filter(timestamp__gte=now - timedelta(days=30)).count()
        prev30 = queries.filter(
            timestamp__gte=now - timedelta(days=60),
            timestamp__lt=now - timedelta(days=30),
        ).count()
        docs_last30 = docs.filter(uploaded_at__gte=now - timedelta(days=30)).count()
        docs_prev30 = docs.filter(
            uploaded_at__gte=now - timedelta(days=60),
            uploaded_at__lt=now - timedelta(days=30),
        ).count()
        ctx["docs_delta"] = _pct_delta(docs_last30, docs_prev30)
        ctx["queries_delta"] = _pct_delta(last30, prev30)

        # response performance
        ctx["avg_response_time"] = round(
            queries.aggregate(x=Avg("response_time_seconds"))["x"] or 0, 1
        )
        succ = (
            queries.filter(status="Responded").count() / queries.count() * 100
            if queries.count() else 0
        )
        ctx["success_rate_pct"] = round(succ, 1)
        esc = (
            queries.filter(status__in=["Escalated", "Failed"]).count()
            / queries.count() * 100
            if queries.count() else 0
        )
        ctx["escalated_pct"] = round(esc, 1)

        # top queried topics
        ctx["top_problems"] = list(
            queries.exclude(problem_entity="")
            .values("problem_entity")
            .annotate(c=Count("id"))
            .order_by("-c")[:8]
        )

        # ----- Monthly trend + drilldown -----
        since = now - timedelta(days=365)
        rows = (
            queries.filter(timestamp__gte=since)
            .annotate(m=TruncMonth("timestamp"))
            .values("m", "category")
            .annotate(c=Count("id"))
            .order_by("m")
        )
        months = {}  # ordered dict: "Mar 2025" -> {"total": x, "cats": {...}}
        for r in rows:
            label = r["m"].strftime("%b %Y") if r["m"] else "—"
            months.setdefault(label, {"total": 0, "cats": defaultdict(int)})
            months[label]["total"] += r["c"]
            months[label]["cats"][r["category"] or "Others"] += r["c"]
        monthly_main = []
        drilldown_series = []
        for label, d in months.items():
            dd_id = "dd_" + label.replace(" ", "_").lower()
            monthly_main.append({"name": label, "y": d["total"], "drilldown": dd_id})
            drilldown_series.append({
                "name": label,
                "id": dd_id,
                "colorByPoint": True,
                "colors": [CATEGORY_COLORS[c] for c in CATEGORY_ORDER],
                "data": [[c, d["cats"].get(c, 0)] for c in CATEGORY_ORDER],
            })

        # ----- District query counts -----
        dist_rows = (
            queries.exclude(district="")
            .values("district")
            .annotate(c=Count("id"))
            .order_by("-c")
        )
        district_counts = {r["district"]: r["c"] for r in dist_rows}

        # Dynamic colour thresholds: five evenly-spaced buckets up to the max.
        # Fallback to a baseline scale if there's no data yet.
        max_count = max(district_counts.values()) if district_counts else 0
        if max_count <= 0:
            thresholds = [1, 2, 3, 4, 5]  # baseline so legend isn't blank
        elif max_count < 5:
            thresholds = [1, 2, 3, 4, max_count]
        else:
            step = max(1, max_count / 5)
            thresholds = [
                max(1, round(step * 1)),
                max(2, round(step * 2)),
                max(3, round(step * 3)),
                max(4, round(step * 4)),
                max_count,
            ]
        # legend labels like "1", "2", "3–4", "5–7", "8+"
        legend = []
        prev = 0
        for i, t in enumerate(thresholds):
            if i == len(thresholds) - 1:
                legend.append(f"{prev + 1 if prev + 1 < t else t}+")
            elif prev + 1 == t:
                legend.append(f"{t}")
            else:
                legend.append(f"{prev + 1}–{t}")
            prev = t

        # ----- Category distribution: totals + per-district -----
        all_cat_totals = _cat_breakdown(queries)

        per_district = {}
        for d_name in list(district_counts.keys())[:24]:
            per_district[d_name] = _cat_breakdown(queries.filter(district=d_name))

        # ----- Recent interactions -----
        ctx["recent_queries"] = queries.order_by("-timestamp")[:5]

        # hooks summary for header
        ctx["active_hooks"] = Hook.objects.filter(status="Active").count()
        ctx["total_hooks"] = Hook.objects.count()

        # district filter choices from the same master used by the heat map
        from core.districts import get_odisha_districts
        ctx["district_choices"] = get_odisha_districts()
        ctx["selected_district"] = self.request.GET.get("district", "")
        ctx["filter_from"] = self.request.GET.get("from", "")
        ctx["filter_to"] = self.request.GET.get("to", "")

        ctx["last_updated"] = now.strftime("%d %b %Y, %H:%M IST")

        ctx["dashboard_json"] = {
            "monthly_main": monthly_main,
            "monthly_drilldown": drilldown_series,
            "district_counts": district_counts,
            "district_thresholds": thresholds,
            "district_legend": legend,
            "category_order": CATEGORY_ORDER,
            "category_colors": [CATEGORY_COLORS[c] for c in CATEGORY_ORDER],
            "all_cat_totals": all_cat_totals,
            "per_district_cat": per_district,
            "total_queries": ctx["kpi_queries"],
        }
        return ctx


def _pct_delta(curr: int, prev: int) -> dict:
    """Return {'text': '↑ 12% vs last month', 'dir': 'up'|'down'|'flat'}"""
    if prev == 0:
        if curr == 0:
            return {"text": "no change vs last month", "dir": "flat"}
        return {"text": "new activity this month", "dir": "up"}
    pct = round((curr - prev) / prev * 100)
    if pct > 0:
        return {"text": f"↑ {pct}% vs last month", "dir": "up"}
    if pct < 0:
        return {"text": f"↓ {abs(pct)}% vs last month", "dir": "down"}
    return {"text": "no change vs last month", "dir": "flat"}


def _cat_breakdown(qs):
    """Return list of ints aligned with CATEGORY_ORDER."""
    counts = Counter(qs.values_list("category", flat=True))
    return [counts.get(c, 0) for c in CATEGORY_ORDER]
