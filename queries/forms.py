from django import forms

from catalogues.models import Catalogue
from core.districts import get_odisha_district_choices


class QueryDemoForm(forms.Form):
    catalogue = forms.ModelChoiceField(
        queryset=Catalogue.objects.filter(vector_db_status="Ready").order_by("name"),
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    farmer_language = forms.ChoiceField(
        choices=[
            ("Odia", "Odia"), ("Hindi", "Hindi"), ("Bengali", "Bengali"),
            ("English", "English"),
        ],
        initial="English",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    district = forms.ChoiceField(
        choices=get_odisha_district_choices(),
        required=False,
        initial="Bargarh",
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    query_text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 3,
            "placeholder": "e.g. How do I treat blast in my rice crop?",
            "style": "padding-right:50px;"
        }),
        help_text="Required unless an image is attached for visual identification.",
    )
    # The farmer's own recording. Populated by the page's recorder (never typed
    # into), and only present when they actually submit — a take they re-recorded
    # because it came out badly is replaced in the browser and never uploaded.
    query_audio = forms.FileField(
        required=False,
        widget=forms.ClearableFileInput(attrs={"accept": "audio/*"}),
        label="Recorded question",
    )
    image = forms.ImageField(
        required=False,
        widget=forms.ClearableFileInput(attrs={"class": "form-control", "accept": "image/*"}),
        label="Or upload an image (optional)",
        help_text="If provided, the system will identify it against the image bank and run a "
                  "full advisory on the matched topic.",
    )

    def clean(self):
        cleaned = super().clean()
        text = (cleaned.get("query_text") or "").strip()
        image = cleaned.get("image")
        if not text and not image:
            raise forms.ValidationError("Provide either a text query or an image.")
        return cleaned
    USER_TYPE_CHOICES = [
        ("farmer", "Farmer — short, plain-language answer"),
        ("researcher", "Researcher — detailed technical answer with citations"),
    ]
    user_type = forms.ChoiceField(
        choices=USER_TYPE_CHOICES,
        initial="farmer",
        widget=forms.RadioSelect,
        label="Ask as",
    )
    generate_tts = forms.BooleanField(
        required=False, initial=False,
        label="Also generate voice response (TTS — costs tokens)",
    )
