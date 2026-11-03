from django import forms

from .models import Keyword


class KeywordForm(forms.ModelForm):
    class Meta:
        model = Keyword
        fields = (
            "keyword", "parent_category", "subcategory", "region",
            "translation", "language", "local_names",
            "expert_validated", "validation_source",
            "image", "status",
        )
        widgets = {
            "keyword": forms.TextInput(attrs={"class": "form-control"}),
            "parent_category": forms.Select(attrs={"class": "form-control"}),
            "subcategory": forms.TextInput(attrs={"class": "form-control"}),
            "region": forms.Select(attrs={"class": "form-control"}),
            "translation": forms.TextInput(attrs={"class": "form-control"}),
            "language": forms.Select(attrs={"class": "form-control"}),
            "local_names": forms.Textarea(attrs={"class": "form-control", "rows": 2}),
            "validation_source": forms.Select(attrs={"class": "form-control"}),
            "status": forms.Select(attrs={"class": "form-control"}),
            "image": forms.ClearableFileInput(attrs={"class": "form-control"}),
        }
