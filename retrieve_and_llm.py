import json
import csv
import re
import os
from langchain_openai import ChatOpenAI
from embeddings import query_embeddings, build_chroma
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import conf

# Define the columns for the JSON data categories and CSV output
columns = [
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


def load_profiles(config_path):
    """Load profiles from an external configuration file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            return json.load(file).get("profiles", [])
    else:
        print(f"Placeholder config file not found: {config_path}")
        return []


profiles = load_profiles(conf.PLACEHOLDER_CONFIG_PATH)

PROMPT_TEMPLATE = """
You are an expert on nutrition. Below is the information you've already provided:
{previous_answers}
 - -
Based only on this context, previous answers and the query:
{context}
 - -
Give nutrition advice based on the above context on this query, only in the form of the actual information: {question}
 - -
Only give suggestions that match the person in the query and give one row per suggestion.
The output has to be in JSON format with the {column} as one key and the source as the other.
Don't include ```json``` in the response.
Don't write multiple answers for percentage of carbs, fat, and protein.
Only use the context that is from the same diet as the query.
Don't format the answer as a list, and don't answer in full sentences, only one word.
Reference all context source metadata that was used.
"""


def format_previous_answers(profile):
    """Formats answers from filled-in columns of the current profile into a readable string for inclusion in the prompt, excluding the source."""
    previous_answers = []

    for category, answers in profile.items():
        if answers:  # Only process if there are answers for the category
            for answer in answers:
                if isinstance(answer, dict):  # Check if 'answer' is a dictionary
                    answer_text = answer.get(category, "")
                    previous_answers.append(f"{category}: {answer_text}")
                else:
                    print(
                        f"Expected a dictionary, but got {type(answer)} for category '{category}'"
                    )
    return "\n".join(previous_answers)


def query_rag(column, profile, previous_answers):
    """Generate and execute a query based on profile-specific values and the query template."""
    query_text = f'Write the {column} column for a lookup table for a {profile["Gender"][0]["Gender"]} aged {profile["Age range"][0]["Age range"]} who wants to {profile["Lose/maintain/gain weight"][0]["Lose/maintain/gain weight"]} using the {profile["Diet name"][0]["Diet name"]}.'

    # Retrieve context using Chroma
    retriever = Chroma(persist_directory=str(conf.CHROMA_PATH))
    results = query_embeddings(conf.CHROMA_PATH, query_text)
    if len(results) == 0:
        print(f"Unable to find matching results for {column}.")
        return None
    context_text = "\n\n - -\n\n".join(
        [f"{doc.page_content} | Metadata: {doc.metadata}" for doc, _score in results]
    )

    # Create prompt template using context, question text, previous answers, and column name
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        previous_answers=previous_answers,
        context=context_text,
        question=query_text,
        column=column,
    )

    # Initialize OpenAI chat model
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # Generate response text based on the prompt
    response_text = model.invoke(prompt)

    return response_text.content.strip()


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


if __name__ == "__main__":
    for index, profile in enumerate(profiles):
        for column in columns:
            if column in profile and profile[column]:  # Skip if already filled
                continue

            # Get previous answers from currently filled columns in the current profile
            previous_answers = format_previous_answers(profile)

            # Generate the query for the current profile and column with previous answers
            result = query_rag(column, profile, previous_answers)
            print(result)
            if result:
                parsed_results = parse_concatenated_json(result)
                if parsed_results:
                    # Ensure profile[column] is a list and append parsed results in the correct format
                    if column not in profile:
                        profile[column] = []

                    for entry in parsed_results:
                        advice = entry.get(column, "")
                        source = entry.get("source", "")
                        full_path_source = build_full_path(source)

                        profile[column].append(
                            {column: advice, "source": full_path_source}
                        )

        # Write profile to JSON and CSV
        json_filename = f"nutrition_advice_{index}.json"
        csv_filename = f"nutrition_advice_{index}.csv"

        with open(json_filename, "w") as json_file:
            json.dump(profile, json_file, indent=4)
        print(f"Data has been written to '{json_filename}'.")

        with open(csv_filename, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_ALL)
            writer.writerow(["Category", "Advice", "Source"])  # Keep Source in headers
            for category, entries in profile.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict):
                            advice = entry.get(category, "")
                            source = entry.get("source", "")

                            # Extract only the part after "content: "
                            content_only_source = re.sub(
                                r"^.*\(content: ", "", source
                            ).rstrip(")")

                            writer.writerow([category, advice, content_only_source])
