"""Live follow-up scenario tests."""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PaddyMitra.settings')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import django; django.setup()

from queries.services.pipeline import run_pipeline_for_text
from queries.services.chat_history import save_chat_turn
from catalogues.models import Catalogue
from pathlib import Path
from django.conf import settings

cat = Catalogue.objects.get(pk=5)

def run_scenario(phone, label, turns):
    hdir = Path(getattr(settings, 'BASE_DIR', '.')) / 'chat_history'
    hfile = hdir / f'{phone}.json'
    if hfile.exists():
        hfile.unlink()

    print(f'\n{"="*60}')
    print(f'SCENARIO: {label}')
    print(f'{"="*60}')
    for i, msg in enumerate(turns):
        print(f'\nTURN {i+1}')
        print(f'USER: {msg}')
        try:
            q = run_pipeline_for_text(cat, 'English', msg,
                                      whatsapp_number=phone)
            ans = q.ai_response_text or '(no answer)'
            print(f'BOT:  {ans}')
            save_chat_turn(phone, msg, ans)
        except Exception as e:
            print(f'ERROR: {e}')
            import traceback; traceback.print_exc()
            break

run_scenario('9999000201', 'Fertilizer -> land type clarification', [
    'How much fertilizer should I apply for early duration rice?',
    'upland',
])

run_scenario('9999000202', 'Production quantity -> area + variety', [
    'How much fertilizer for 1900 tons of rice?',
    '2 hectares irrigated MTU 1010',
])

run_scenario('9999000203', 'Variety list -> which one (entity ref follow-up)', [
    'Which rice variety is suitable for lowland?',
    'Which one matures fastest?',
])

run_scenario('9999000204', 'Topic change: urea -> variety', [
    'How much urea per acre?',
    'What is the best variety for upland?',
])

print('\n\nDone.')
