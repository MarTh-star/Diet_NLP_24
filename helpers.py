from pathlib import Path
import json
import re
import csv

import conf


def load_profiles(config_path: Path) -> list[dict]:
    """Load profiles from an external configuration file."""
    if config_path.exists():
        with config_path.open("r") as file:
            return json.load(file).get("profiles", [])
    else:
        print(f"Placeholder config file not found: {config_path}")
        return []


def parse_concatenated_json(response_text):
    """Parse concatenated JSON objects by recognizing complete objects individually.

    This is to handle when the response contains multiple JSON objects but they
    are not always formatted exactly alike."""
    parsed_data = []
    current_object = ""
    open_braces = 0  # Track open braces to identify complete JSON objects

    for line in response_text.splitlines():
        line = line.strip()

        if line.startswith("{"):
            open_braces += 1
        if line.endswith("}"):
            open_braces -= 1

        # Accumulate lines within a JSON object
        current_object += line + " "

        # If we've closed all open braces, it's a complete JSON object
        if open_braces == 0 and current_object.strip():
            try:
                parsed_data.append(json.loads(current_object))
            except json.JSONDecodeError:
                print("Failed to parse JSON object:", current_object)
            finally:
                # Reset for the next JSON object
                current_object = ""

    return parsed_data


def build_full_path(source: str) -> str:
    """Convert the source text to a full file path, dynamically using the diet directory name and numeric filename."""
    match = re.match(r"^(.*?)_(\d+)\.csv_(.*)$", source)
    if match:
        diet_directory = match.group(1)
        numeric_filename = f"{match.group(2)}.csv"
        suffix = match.group(3)
        full_path = conf.DATA_PATH / diet_directory / numeric_filename
        try:
            with open(full_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    if suffix in row:
                        return f"{full_path}_{suffix} (content: {row})"
        except FileNotFoundError:
            print(f"File not found: {full_path}")
        except Exception as e:
            print(f"Error reading file {full_path}: {e}")
        return f"{full_path}_{suffix}"
    else:
        return source


def init_advice_directories():
    """Create directories for storing advice data, if not present."""
    if not conf.NUTRITION_ADVICE_CSV_DIR.exists():
        conf.NUTRITION_ADVICE_CSV_DIR.mkdir(exist_ok=True)

    if not conf.NUTRITION_ADVICE_JSON_DIR.exists():
        conf.NUTRITION_ADVICE_JSON_DIR.mkdir(exist_ok=True)
