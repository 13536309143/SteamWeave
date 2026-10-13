from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env', override=False)

def local_path(name, default):
    p = Path(os.getenv(name, default)).expanduser()
    return p if p.is_absolute() else ROOT / p

DATASET = local_path('DATASET_PATH', 'Multimodal data for prognostics')
ARTIFACTS = local_path('ARTIFACT_PATH', 'artifacts')
ARTIFACTS.mkdir(parents=True, exist_ok=True)
MODEL = os.getenv('MINIMAX_MODEL', 'MiniMax-M2.7')
BASE_URL = os.getenv('MINIMAX_BASE_URL', 'https://api.minimax.cn/v1').rstrip('/')
