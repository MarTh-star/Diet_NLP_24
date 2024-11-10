# Diet_NLP_24

Welcome to the Diet_NLP_24 project! This repository contains code and resources for natural language processing (NLP) tasks related to dietary information.

## Table of Contents

- [Diet\_NLP\_24](#diet_nlp_24)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Python](#python)
  - [Installation](#installation)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Main Scripts](#main-scripts)
    - [conf](#conf)
      - [Overview](#overview)
      - [Configuration details](#configuration-details)
    - [embeddings](#embeddings)
      - [Overview](#overview-1)
      - [Main functions](#main-functions)
    - [retrieve\_and\_llm](#retrieve_and_llm)
      - [Overview](#overview-2)
      - [Main functions](#main-functions-1)
  - [Project Structure](#project-structure)
  - [Example Patient Profile](#example-patient-profile)

## Introduction

Diet_NLP_24 is a project aimed at leveraging an RAG model to analyze and process dietary information. The goal is to provide a RAG model that can help fill in a lookup table with dietary data based on a patient profile and scientific resources.

## Python

During development `python 3.12.7` has been used and is recommended to guarantee that everything functions.

## Installation

To get started with this project, clone the repository and install the required dependencies, it is recommended to create a new environment first:

```bash
git clone https://github.com/MarTh-star/Diet_NLP_24.git
cd Diet_NLP_24
pip install -r requirements.txt
```

## Setup

Create a `.env` file and add your `OPENAI_API_KEY`.  
After installing the dependencies and adding your openai api key, you have to start embedding the dataset and putting it in the vector store.  
You can do this by running:

```bash
python embeddings.py
```

Which will take the values in the placeholders.json file and retrieve

## Usage

For the purposes of this project there is already a `placeholders.json`file which contain example profiles that can be used. These are retrieved automatically when running the script below.

Once the setup is finished you can run the script:

```bash
python retrieve_and_llm.py
```

## Main Scripts

### conf

#### Overview

The `conf.py` file serves as the configuration hub for your project, defining essential paths and secure settings. It is crucial for setting up the environment and integrating with external APIs.

#### Configuration details

- `BASE_PATH`: Automatically set to the parent directory of the conf.py file.
- `DATA_PATH`: Specifies the directory path for storing data files; defaults to a directory named diet_data under BASE_PATH.
- `CHROMA_PATH`: Specifies the directory path for storing Chroma database files; defaults to a directory named chroma_data under BASE_PATH.
- `OPENAI_API_KEY`: A secure way to handle the OpenAI API key using pydantic's SecretStr type. It expects the key to be provided via environment variables.

### embeddings

#### Overview

The `embeddings.py` script is designed to handle document embeddings and retrieval for datasets. It utilizes OpenAI's embeddings through the `langchain_openai` and `langchain_chroma` packages to process, index, and query documents efficiently.

#### Main functions

- `load_documents(data_path)`: Loads documents from CSV files in the specified directory, cleaning and structuring them into a list of Document objects.
- `save_to_chroma(documents, chroma_path, batch_size)`: Saves processed documents into a Chroma database for later querying, handling batches of documents to manage memory and performance effectively.
- `build_chroma(data_path, chroma_path, overwrite)`: Processes an entire dataset from data_path, saving the embeddings to chroma_path. If overwrite is True, it will clear the existing data at chroma_path before proceeding.
- `query_embeddings(chroma_path, query, top_k, threshold)`: Performs a query against the Chroma database and returns relevant documents based on similarity scores.

### retrieve_and_llm

#### Overview

This script processes nutrition advice based on user profiles loaded from a JSON configuration file. It leverages various components from `langchain_openai` and local modules to query, process, and format advice into JSON and CSV formats.

#### Main functions

- `load_profiles`: Loads nutrition profiles from a configuration file.
- `query_rag`: Performs queries using retrieval-augmented generation to fetch relevant advice.
- `parse_concatenated_json`: Parses concatenated JSON strings into structured data.
- `format_previous_answers`: Formats previously provided answers to be used in prompts.

## Project Structure

The directory structure of the project is as follows:

```
├── README.md
├── conf.py                 # Script to read
├── diet_data               # Dataset with chunked papers
│   ├── dash diet
│   ├── intermittent fasting
│   ├── ketogenic diet
│   ├── mediterranean diet
│   └── nordic diet
├── nutrition_advice_csv    # Example output using user profiles in placeholders.json file
├── nutrition_advice_json   # Example output using user profiles in placeholders.json file
├── embeddings.py           # Script for embedding the dataset
├── placeholders.json       # Example patient profiles that is used for the RAG model
├── requirements.txt        # Requirements that should be installed in order to run the project
└── retrieve_and_llm.py     # The script for prompting gpt-4o and formatting the output and saving it in csv and json files.
```

## Example Patient Profile

The patient profiles used for the program follow this structure:

```json
{
  "Age range": [{ "Age range": "26-35 years", "source": "pre-filled" }],
  "Gender": [{ "Gender": "Male", "source": "pre-filled" }],
  "Lose/maintain/gain weight": [
    { "Lose/maintain/gain weight": "Lose weight", "source": "pre-filled" }
  ],
  "Diet name": [{ "Diet name": "dash diet", "source": "pre-filled" }],
  "Health Pre-Condition": [
    { "Health Pre-Condition": "High blood pressure", "source": "pre-filled" }
  ],
  "Foods to increase consumption of": [],
  "Foods to eat in moderation": [],
  "Foods to avoid": [],
  "Macros: Percent of Fat": [],
  "Percent of Protein": [],
  "Percent of Carbs": []
}
```

The patient data as well as the diet is pre-filled and the RAG model is then tasked with filling in the food and the macros information.
