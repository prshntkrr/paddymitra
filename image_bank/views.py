from __future__ import annotations

import logging
from pathlib import Path

from django.contrib import messages
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.generic import View

from catalogues.models import Catalogue
from core.mixins import RoleRequiredMixin

from .forms import GroupDescriptionForm
from .models import ImageGroup, KnowledgeImage
from .services.image_vector_store import (
    add_image as vs_add,
    count as vs_count,
    purge_catalogue as vs_purge_catalogue,
    remove_group as vs_remove_group,
    remove_image as vs_remove,
    reset_collection as vs_reset,
)

log = logging.getLogger("irri")

EDITOR_ROLES = ("Administrator", "Editor")


# ----------------- Catalogue picker (top-level) -----------------

class CataloguePickerView(RoleRequiredMixin, View):
    required_roles = EDITOR_ROLES
    template_name = "image_bank/catalogue_picker.html"

    def get(self, request):
        cats = Catalogue.objects.annotate(
            n_groups=Count("image_groups", distinct=True),
            n_images=Count("knowledge_images", distinct=True),
        ).order_by("name")
        return render(request, self.template_name, {
            "catalogues": cats,
            "total_images": KnowledgeImage.objects.count(),
            "total_groups": ImageGroup.objects.count(),
            "vector_count": vs_count(),
        })


# ----------------- Group list (per catalogue) -----------------

class GroupListView(RoleRequiredMixin, View):
    required_roles = EDITOR_ROLES
    template_name = "image_bank/group_list.html"

    def get(self, request, slug):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        groups = (
            catalogue.image_groups
            .annotate(n_images=Count("images"))
            .order_by("prefix")
        )
        return render(request, self.template_name, {
            "catalogue": catalogue,
            "groups": groups,
            "total_groups": groups.count(),
            "total_images": KnowledgeImage.objects.filter(catalogue=catalogue).count(),
            "vector_count": vs_count(catalogue_id=catalogue.id),
        })


# ----------------- Upload / create new group -----------------

class GroupCreateView(RoleRequiredMixin, View):
    """Bulk upload: sidecar(s) + image(s). One sidecar = one group."""
    required_roles = EDITOR_ROLES
    template_name = "image_bank/group_create.html"

    def get(self, request, slug):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        return render(request, self.template_name, {"catalogue": catalogue})

    def post(self, request, slug):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        sidecars = request.FILES.getlist("sidecars")
        image_files = request.FILES.getlist("images")

        if not sidecars or not image_files:
            messages.error(request, "Please upload at least one sidecar .txt/.md file and at least one image.")
            return render(request, self.template_name, {"catalogue": catalogue})

        # Build sidecar table { stem_lower: (stem_original, description) }
        sidecar_table = {}
        for sf in sidecars:
            stem = Path(sf.name).stem
            try:
                text = sf.read().decode("utf-8", errors="ignore").strip()
            except Exception as exc:
                messages.warning(request, f"Could not read {sf.name}: {exc}")
                continue
            sidecar_table[stem.lower()] = (stem, text)

        if not sidecar_table:
            messages.error(request, "Sidecar files were unreadable.")
            return render(request, self.template_name, {"catalogue": catalogue})

        # Ensure ImageGroup exists for every sidecar (even if no image matches yet).
        groups_by_stem_lower = {}
        for stem_lower, (stem_original, description) in sidecar_table.items():
            grp, _ = ImageGroup.objects.update_or_create(
                catalogue=catalogue, prefix=stem_original,
                defaults={"description": description},
            )
            # If this group already had images and the description changed, the per-image
            # `description` mirror will get refreshed on the next add (or via re-index).
            groups_by_stem_lower[stem_lower] = grp

        created = 0
        indexed = 0
        unmatched = []
        groups_used = set()
        for f in image_files:
            base_lower = Path(f.name).stem.lower()
            best_stem_lower = ""
            for stem_lower in sidecar_table.keys():
                if (
                    base_lower == stem_lower
                    or base_lower.startswith(stem_lower + "_")
                    or base_lower.startswith(stem_lower + "-")
                    or stem_lower in base_lower.split("_")
                    or stem_lower in base_lower.split("-")
                ):
                    if len(stem_lower) > len(best_stem_lower):
                        best_stem_lower = stem_lower
            if not best_stem_lower:
                unmatched.append(f.name)
                continue
            grp = groups_by_stem_lower[best_stem_lower]
            ki = KnowledgeImage.objects.create(
                group=grp,
                catalogue=catalogue,
                prefix=grp.prefix,
                image=f,
                original_filename=f.name,
                description=grp.description,
            )
            created += 1
            try:
                vs_add(ki)
                indexed += 1
            except Exception as exc:
                log.exception("CLIP indexing failed for %s", ki.pk)
                messages.warning(request, f"Saved {f.name} but indexing failed: {exc}")
            groups_used.add(grp.prefix)

        if unmatched:
            messages.warning(
                request,
                f"Skipped {len(unmatched)} image(s) with no matching sidecar prefix: "
                f"{', '.join(unmatched[:5])}{' …' if len(unmatched) > 5 else ''}",
            )
        if created:
            messages.success(
                request,
                f"Uploaded {created} image(s) across {len(groups_used)} group(s) "
                f"({', '.join(sorted(groups_used))}). {indexed} indexed in CLIP store.",
            )
        return redirect(f"{reverse('catalogues:detail', kwargs={'slug': catalogue.slug})}#image-groups")


# ----------------- Group detail: images, add more, edit description -----------------

class GroupDetailView(RoleRequiredMixin, View):
    required_roles = EDITOR_ROLES
    template_name = "image_bank/group_detail.html"

    def get(self, request, slug, gid):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        group = get_object_or_404(ImageGroup, pk=gid, catalogue=catalogue)
        return render(request, self.template_name, {
            "catalogue": catalogue,
            "group": group,
            "images": group.images.all(),
            "desc_form": GroupDescriptionForm(instance=group),
        })


class GroupAddImagesView(RoleRequiredMixin, View):
    required_roles = EDITOR_ROLES

    def post(self, request, slug, gid):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        group = get_object_or_404(ImageGroup, pk=gid, catalogue=catalogue)
        files = request.FILES.getlist("images")
        if not files:
            messages.error(request, "No images selected.")
            return redirect("image_bank:group_detail", slug=catalogue.slug, gid=group.pk)
        added = 0
        indexed = 0
        for f in files:
            ki = KnowledgeImage.objects.create(
                group=group,
                catalogue=catalogue,
                prefix=group.prefix,
                image=f,
                original_filename=f.name,
                description=group.description,
            )
            added += 1
            try:
                vs_add(ki)
                indexed += 1
            except Exception as exc:
                log.exception("Indexing failed for %s", ki.pk)
                messages.warning(request, f"Saved {f.name} but indexing failed: {exc}")
        messages.success(request, f"Added {added} image(s) to group '{group.prefix}'. {indexed} indexed.")
        return redirect("image_bank:group_detail", slug=catalogue.slug, gid=group.pk)


class GroupUpdateDescriptionView(RoleRequiredMixin, View):
    required_roles = EDITOR_ROLES

    def post(self, request, slug, gid):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        group = get_object_or_404(ImageGroup, pk=gid, catalogue=catalogue)

        # Optional: replace description from an uploaded sidecar file too.
        new_desc = request.POST.get("description", "").strip()
        sidecar = request.FILES.get("sidecar")
        if sidecar is not None:
            try:
                new_desc = sidecar.read().decode("utf-8", errors="ignore").strip() or new_desc
            except Exception as exc:
                messages.warning(request, f"Could not read sidecar file: {exc}")

        group.description = new_desc
        group.save(update_fields=["description", "updated_at"])

        # Re-index every image in this group with the new description as document text.
        reindexed = 0
        for ki in group.images.all():
            ki.description = new_desc
            try:
                vs_add(ki)
                reindexed += 1
            except Exception as exc:
                log.warning("Re-index failed for %s: %s", ki.pk, exc)
        messages.success(request, f"Description updated. Re-indexed {reindexed} image(s).")
        return redirect("image_bank:group_detail", slug=catalogue.slug, gid=group.pk)


class GroupDeleteView(RoleRequiredMixin, View):
    required_roles = EDITOR_ROLES
    template_name = "image_bank/group_confirm_delete.html"

    def get(self, request, slug, gid):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        group = get_object_or_404(ImageGroup, pk=gid, catalogue=catalogue)
        return render(request, self.template_name, {"catalogue": catalogue, "group": group})

    def post(self, request, slug, gid):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        group = get_object_or_404(ImageGroup, pk=gid, catalogue=catalogue)
        prefix = group.prefix
        # Remove individual rows by id (catches normal entries) AND by metadata
        # (catches orphaned vectors whose ki_id no longer exists, e.g. left over
        # from a previous prefix that has since been deleted).
        for ki in group.images.all():
            vs_remove(ki.pk)
        vs_remove_group(group.pk, prefix=prefix)
        group.delete()
        messages.success(
            request,
            f"Group '{prefix}' deleted along with all its CLIP vectors.",
        )
        return redirect(f"{reverse('catalogues:detail', kwargs={'slug': catalogue.slug})}#image-groups")


class ImageDeleteView(RoleRequiredMixin, View):
    required_roles = EDITOR_ROLES

    def post(self, request, slug, gid, ki_id):
        catalogue = get_object_or_404(Catalogue, slug=slug)
        group = get_object_or_404(ImageGroup, pk=gid, catalogue=catalogue)
        ki = get_object_or_404(KnowledgeImage, pk=ki_id, group=group)
        vs_remove(ki.pk)
        ki.delete()
        messages.success(request, "Image deleted.")
        return redirect("image_bank:group_detail", slug=catalogue.slug, gid=group.pk)


# ----------------- Reindex catalogue -----------------

def reindex_catalogue(request, slug):
    role = getattr(request.user, "role", None)
    role_name = role.name if role else ""
    if role_name not in ("Administrator",):
        messages.error(request, "Only administrators can reindex.")
        return redirect("image_bank:groups", slug=slug)
    catalogue = get_object_or_404(Catalogue, slug=slug)
    # First purge every vector belonging to this catalogue (kills orphans).
    vs_purge_catalogue(catalogue.id)
    n = 0
    for ki in KnowledgeImage.objects.filter(catalogue=catalogue):
        try:
            vs_add(ki)
            n += 1
        except Exception:
            log.exception("Reindex failed for %s", ki.pk)
    messages.success(
        request,
        f"Purged old vectors and reindexed {n} image(s) in '{catalogue.name}'.",
    )
    return redirect("image_bank:groups", slug=catalogue.slug)


def purge_all_vectors(request):
    """Admin-only: wipe the entire `image_bank` Chroma collection (all catalogues),
    then re-embed every KnowledgeImage that exists in the DB. Use this if stale
    vectors from previously-deleted images are still influencing answers."""
    role = getattr(request.user, "role", None)
    role_name = role.name if role else ""
    if role_name not in ("Administrator",):
        messages.error(request, "Only administrators can purge image vectors.")
        return redirect("image_bank:list")
    vs_reset()
    n = 0
    for ki in KnowledgeImage.objects.all():
        try:
            vs_add(ki)
            n += 1
        except Exception:
            log.exception("Re-add failed for %s", ki.pk)
    messages.success(
        request,
        f"Purged the entire image vector store and re-indexed {n} image(s) from the DB.",
    )
    return redirect("image_bank:list")
