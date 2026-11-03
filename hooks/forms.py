from django import forms

from catalogues.models import Catalogue

from .models import Hook


class HookForm(forms.ModelForm):
    catalogue = forms.ModelChoiceField(
        queryset=Catalogue.objects.all().order_by("name"),
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    class Meta:
        model = Hook
        fields = (
            "name", "whatsapp_number", "gateway_provider", "trigger_keyword",
            "catalogue", "primary_language", "secondary_language",
            "stt_provider", "tts_provider", "translation_model", "rag_model",
            "status", "welcome_message", "webhook_url",
        )
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control"}),
            "whatsapp_number": forms.TextInput(attrs={"class": "form-control"}),
            "gateway_provider": forms.Select(attrs={"class": "form-control"}),
            "trigger_keyword": forms.TextInput(attrs={"class": "form-control"}),
            "primary_language": forms.Select(attrs={"class": "form-control"}),
            "secondary_language": forms.Select(attrs={"class": "form-control"}),
            "stt_provider": forms.TextInput(attrs={"class": "form-control"}),
            "tts_provider": forms.TextInput(attrs={"class": "form-control"}),
            "translation_model": forms.TextInput(attrs={"class": "form-control"}),
            "rag_model": forms.TextInput(attrs={"class": "form-control"}),
            "status": forms.Select(attrs={"class": "form-control"}),
            "welcome_message": forms.Textarea(attrs={"class": "form-control", "rows": 2}),
            "webhook_url": forms.TextInput(attrs={"class": "form-control"}),
        }
