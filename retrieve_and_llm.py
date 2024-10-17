from langchain_openai import ChatOpenAI
from embeddings import query_embeddings, build_chroma
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import conf
import getpass
import os
import re

# query_text = "30 years old, gender woman, bmi 21.9, lose weight."
query_text = "I am a 30 year old man, with a bmi of 21.9 trying to gain weight."
# columns = "Foods to increase consumption of, Foods to eat in moderation, Foods to avoid, (For intermittent fasting) Eating time window, Duration (numeric - in weeks) (for intermittent fasting), Macros: Percent of Fat, Percent of Protein, Percent of Carbs"
columns = "Foods to increase consumption of, Foods to eat in moderation, Foods to avoid, Macros: Percent of Fat, Percent of Protein, Percent of Carbs"


PROMPT_TEMPLATE = """
You are an expert on diets, give dietary adviced based only on this context:
{context}
 - -
Give dietary advice based on the above context: {question}
 - -
The advice should be in the form of a csv that is delimited with a semi-colon table that have the following columns: {columns}
Chose one diet out of dash, intermittent fasting, ketogenic, mediterranean or nordic diet and if intermittent fasting is recommended give the time window and duration of the fasting.
Write the percentage of macros, fat and carbs only once.
List the metadata of the context text and which diet was chosen.
"""

def query_rag(query_text):
    retriever = Chroma(persist_directory=str(conf.CHROMA_PATH))
    results = query_embeddings(conf.CHROMA_PATH, query_text)

    for res in results:
        print(res[0].metadata)

    if len(results) == 0:
        print(f"Unable to find matching results.")
        return
    context_text = "\n\n - -\n\n".join(
    [f"{doc.page_content} | Metadata: {doc.metadata}" for doc, _score in results])

    print(context_text)

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, columns=columns)
  
    # Initialize OpenAI chat model
    model = ChatOpenAI(model="gpt-4o-mini")

    # Generate response text based on the prompt
    response_text = model.predict(prompt)

    return response_text

if __name__ == "__main__":
    build_chroma(data_path=conf.DATA_PATH, chroma_path=conf.CHROMA_PATH, overwrite=False)
    print(query_rag(query_text))