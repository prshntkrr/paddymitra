from django import forms

from keywords.models import Keyword

from .models import Category, Document, Subcategory


class DocumentForm(forms.ModelForm):
    keywords = forms.ModelMultipleChoiceField(
        queryset=Keyword.objects.filter(status=Keyword.STATUS_PUBLISHED).order_by("keyword"),
        required=False,
        widget=forms.SelectMultiple(attrs={"class": "form-control", "size": 6}),
    )
    category = forms.ModelChoiceField(
        queryset=Category.objects.all(),
        required=False,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    subcategory = forms.ModelChoiceField(
        queryset=Subcategory.objects.all(),
        required=False,
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    class Meta:
        model = Document
        fields = (
            "title", "file", "crop", "content_type", "doc_type",
            "category", "subcategory", "keywords",
            "geography", "year", "source", "source_url", "summary",
            "authors", "organizations", "journal_or_book", "countries",
        )
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control"}),
            "crop": forms.Select(attrs={"class": "form-control"}),
            "content_type": forms.Select(attrs={"class": "form-control"}),
            "doc_type": forms.Select(attrs={"class": "form-control"}),
            "geography": forms.TextInput(attrs={"class": "form-control"}),
            "year": forms.NumberInput(attrs={"class": "form-control"}),
            "source": forms.TextInput(attrs={"class": "form-control"}),
            "source_url": forms.URLInput(attrs={"class": "form-control"}),
            "summary": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
            "authors": forms.Textarea(attrs={"class": "form-control", "rows": 2}),
            "organizations": forms.Textarea(attrs={"class": "form-control", "rows": 2}),
            "journal_or_book": forms.TextInput(attrs={"class": "form-control"}),
            "countries": forms.TextInput(attrs={"class": "form-control"}),
            "file": forms.ClearableFileInput(attrs={
                "class": "form-control",
                "accept": ".pdf,.docx,.xlsx,.csv,.txt",
            }),
        }

    ALLOWED_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".csv", ".txt")

    def clean_file(self):
        f = self.cleaned_data.get("file")
        if f and getattr(f, "name", ""):
            name_lower = f.name.lower()
            if not any(name_lower.endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
                raise forms.ValidationError(
                    "Only PDF, DOCX, XLSX, CSV, or TXT files can be uploaded."
                )
        return f
