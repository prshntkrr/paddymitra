from django import forms

from documents.models import Document

from .models import Catalogue


class CatalogueForm(forms.ModelForm):
    documents = forms.ModelMultipleChoiceField(
        queryset=Document.objects.filter(status="Published").order_by("title"),
        required=False,
        widget=forms.SelectMultiple(attrs={"class": "form-control", "size": 10}),
    )

    class Meta:
        model = Catalogue
        fields = (
            "name", "slug", "description",
            "geography", "season_year", "status", "documents",
        )
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "slug": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "auto-generated from name if blank",
            }),
            "description": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
            "geography": forms.TextInput(attrs={"class": "form-control"}),
            "season_year": forms.TextInput(attrs={"class": "form-control"}),
            "status": forms.Select(attrs={"class": "form-control"}),
        }
