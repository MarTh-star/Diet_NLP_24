import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# Paths and files
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "diet_data"
CHROMA_PATH = BASE_PATH / "chroma_data"
PROGRESS_LOG = BASE_PATH / "progress.log"
PLACEHOLDER_CONFIG_PATH = BASE_PATH / "placeholders.json"
NUTRITION_ADVICE_CSV_DIR = BASE_PATH / "nutrition_advice_csv"
NUTRITION_ADVICE_JSON_DIR = BASE_PATH / "nutrition_advice_json"

# OpenAI model related constants
OPENAI_API_KEY = SecretStr(os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o"

# RAG config
OUTPUT_COLUMNS = [
    "Age range",
    "Gender",
    "Lose/maintain/gain weight",
    "Diet name",
    "Health Pre-Condition",
    "Foods to increase consumption of",
    "Foods to eat in moderation",
    "Foods to avoid",
    "Macros: Percent of Fat",
    "Percent of Protein",
    "Percent of Carbs",
]
