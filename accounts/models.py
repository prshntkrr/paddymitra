from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models


class Role(models.Model):
    ADMINISTRATOR = "Administrator"
    EDITOR = "Editor"
    KNOWLEDGE_WORKER = "Knowledge Worker"
    REVIEWER = "Reviewer"
    ROLE_CHOICES = [
        (ADMINISTRATOR, "Administrator"),
        (EDITOR, "Editor"),
        (KNOWLEDGE_WORKER, "Knowledge Worker"),
        (REVIEWER, "Reviewer"),
    ]

    name = models.CharField(max_length=64, unique=True, choices=ROLE_CHOICES)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class UserManager(BaseUserManager):
    """Email-based user manager."""

    use_in_migrations = True

    def _create_user(self, email, password, **extra):
        if not email:
            raise ValueError("Email is required")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra):
        extra.setdefault("is_staff", False)
        extra.setdefault("is_superuser", False)
        return self._create_user(email, password, **extra)

    def create_superuser(self, email, password=None, **extra):
        extra.setdefault("is_staff", True)
        extra.setdefault("is_superuser", True)
        if extra.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")
        return self._create_user(email, password, **extra)


class User(AbstractUser):
    """
    Custom user: email is the login field.
    `username` is still present (required by AbstractUser) but not used for login.
    """

    ORG_IRRI = "IRRI"
    ORG_PRIVATE = "Private Company"
    ORG_NGO = "NGO"
    ORG_KVK = "KVK"
    ORG_UNIVERSITY = "University"
    ORG_GOVT = "Government"
    ORG_OTHER = "Other"
    ORG_CHOICES = [
        (ORG_IRRI, "IRRI"),
        (ORG_PRIVATE, "Private Company"),
        (ORG_NGO, "NGO"),
        (ORG_KVK, "KVK"),
        (ORG_UNIVERSITY, "University"),
        (ORG_GOVT, "Government"),
        (ORG_OTHER, "Other"),
    ]

    STATUS_ACTIVE = "Active"
    STATUS_INACTIVE = "Inactive"
    STATUS_CHOICES = [(STATUS_ACTIVE, "Active"), (STATUS_INACTIVE, "Inactive")]

    email = models.EmailField(unique=True)
    full_name = models.CharField(max_length=150, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    role = models.ForeignKey(
        Role, on_delete=models.PROTECT, related_name="users", null=True, blank=True
    )
    organization_type = models.CharField(
        max_length=40, choices=ORG_CHOICES, default=ORG_IRRI
    )
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=STATUS_ACTIVE)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    objects = UserManager()

    class Meta:
        ordering = ["email"]

    def __str__(self) -> str:
        return self.full_name or self.email

    @property
    def display_name(self) -> str:
        return self.full_name or self.email.split("@")[0]

    @property
    def initials(self) -> str:
        if self.full_name:
            parts = self.full_name.split()
            if len(parts) >= 2:
                return (parts[0][0] + parts[-1][0]).upper()
            return parts[0][:2].upper()
        return self.email[:2].upper()
