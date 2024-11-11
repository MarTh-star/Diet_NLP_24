import json
import csv
import re

from langchain_openai import ChatOpenAI
from embeddings import query_embeddings
from langchain_core.prompts import ChatPromptTemplate
import conf

from helpers import (
    load_profiles,
    parse_concatenated_json,
    build_full_path,
    init_advice_directories,
)

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


def format_previous_answers(profile: dict) -> str:
    """Formats answers from filled-in columns of the current profile into a readable string for inclusion in the prompt, excluding the source."""
    previous_answers = []

    for category, answers in profile.items():
        if answers:  # Only process if there are answers for the category
            for answer in answers:
                if isinstance(answer, dict):
                    answer_text = answer.get(category, "")
                    previous_answers.append(f"{category}: {answer_text}")
                else:
                    print(
                        f"Expected a dictionary, but got {type(answer)} for category '{category}'"
                    )
    return "\n".join(previous_answers)


def query_rag(column: str, profile: dict, previous_answers: str) -> str | None:
    """Generate and execute a query based on profile-specific values and the query template."""
    query_text = f'Write the {column} column for a lookup table for a {profile["Gender"][0]["Gender"]} aged {profile["Age range"][0]["Age range"]} who wants to {profile["Lose/maintain/gain weight"][0]["Lose/maintain/gain weight"]} using the {profile["Diet name"][0]["Diet name"]}.'

    # Retrieve context using Chroma
    results = query_embeddings(conf.CHROMA_PATH, query_text)
    if len(results) == 0:
        print(f"Unable to find matching results for {column}.")
        return None
    context_text = "\n\n - -\n\n".join(
        [f"{doc.page_content} | Metadata: {doc.metadata}" for doc in results]
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
    model = ChatOpenAI(model=conf.MODEL, temperature=0)

    # Generate response text based on the prompt
    response_text = model.invoke(prompt)

    return response_text.content.strip()  # type: ignore


def generate_based_on_user_profile(profile: dict, column: str) -> dict | None:
    """Fill output columns based on the user profile."""
    # Get previous answers from currently filled columns in the current profile
    previous_answers = format_previous_answers(profile)

    # Generate the query for the current profile and column with previous answers
    result = query_rag(column, profile, previous_answers)
    print(result)
    if result:
        parsed_results = parse_concatenated_json(result)
        if parsed_results:
            for entry in parsed_results:
                advice = entry.get(column, "")
                source = entry.get("source", "")
                full_path_source = build_full_path(source)

                return {column: advice, "source": full_path_source}


def store_profile_to_file(profile: dict, idx: int) -> None:
    """Saves profile to JSON and CSV files."""
    json_file = conf.NUTRITION_ADVICE_JSON_DIR / f"nutrition_advice_{idx}.json"
    csv_file = conf.NUTRITION_ADVICE_CSV_DIR / f"nutrition_advice_{idx}.csv"

    with json_file.open("w") as json_file:
        json.dump(profile, json_file, indent=4)

    with csv_file.open("w", newline="") as csv_file:
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

    print(f"Patient profile with index={idx} saved successfully.")


if __name__ == "__main__":
    init_advice_directories()

    profiles = load_profiles(conf.PLACEHOLDER_CONFIG_PATH)

    for idx, profile in enumerate(profiles):
        for column in conf.OUTPUT_COLUMNS:
            # Skip if already filled
            if column in profile and profile[column]:
                continue

            # Add column if not existing
            if column not in profile:
                profile[column] = []

            profile[column].append(generate_based_on_user_profile(profile, column))

        store_profile_to_file(profile, idx)
