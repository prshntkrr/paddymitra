"""
Creates the 4 canonical roles and 4 seed users (password: qwerty@12).
Idempotent.
"""
from django.core.management.base import BaseCommand

from accounts.models import Role, User


SEED_USERS = [
    ("admin@irri.local", "Admin User", "Administrator", "IRRI"),
    ("editor@irri.local", "Rajesh Patel", "Editor", "IRRI"),
    ("kw@irri.local", "Suman KW", "Knowledge Worker", "KVK"),
    ("reviewer@irri.local", "Amit Reviewer", "Reviewer", "IRRI"),
]
SEED_PASSWORD = "qwerty@12"


class Command(BaseCommand):
    help = "Seed the 4 roles and 4 demo users (password: qwerty@12)."

    def handle(self, *args, **options):
        roles_desc = {
            "Administrator": "Full system access",
            "Editor": "Content management — documents, keywords, catalogues",
            "Knowledge Worker": "Upload & draft only; needs Reviewer approval",
            "Reviewer": "Approve / reject knowledge worker submissions; flag queries",
        }
        created_roles = 0
        for name, desc in roles_desc.items():
            _, created = Role.objects.get_or_create(name=name, defaults={"description": desc})
            created_roles += int(created)
        self.stdout.write(self.style.SUCCESS(f"Roles: {created_roles} created, {len(roles_desc) - created_roles} already existed."))

        for email, name, role_name, org in SEED_USERS:
            role = Role.objects.get(name=role_name)
            user, created = User.objects.get_or_create(
                email=email,
                defaults={
                    "username": email.split("@")[0],
                    "full_name": name,
                    "role": role,
                    "organization_type": org,
                    "is_staff": role_name == "Administrator",
                    "is_superuser": role_name == "Administrator",
                    "status": "Active",
                },
            )
            if created or not user.has_usable_password():
                user.set_password(SEED_PASSWORD)
                user.save()
            verb = "created" if created else "updated"
            self.stdout.write(f"  {verb}: {email} ({role_name})")
        self.stdout.write(self.style.SUCCESS("Seed users ready. Password: qwerty@12"))
