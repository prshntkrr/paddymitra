from django import forms

from .models import ImageGroup


class GroupCreateForm(forms.Form):
    """Create a new visual group from a sidecar .txt + a batch of images.
    The sidecar's filename (without extension) becomes the group's prefix and
    its contents the visual description."""

    # The actual sidecar/images files are read from request.FILES.getlist(...)
    # in the view because Django's ClearableFileInput refuses multiple=True.

    def clean(self):
        return super().clean()


class GroupDescriptionForm(forms.ModelForm):
    """Edit a group's visual description (the sidecar text)."""

    class Meta:
        model = ImageGroup
        fields = ["description"]
        widgets = {
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 8}),
        }


class AddImagesToGroupForm(forms.Form):
    """Add more images to an existing group. Multiple files via raw template input."""

    def clean(self):
        return super().clean()
