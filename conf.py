import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "diet_data"
CHROMA_PATH = BASE_PATH / "chroma_data"

PROGRESS_LOG = "progress.log"

OPENAI_API_KEY = SecretStr(os.environ["OPENAI_API_KEY"])

BASE_DIRECTORY = "diet_data"
PLACEHOLDER_CONFIG_PATH = "placeholders.json"