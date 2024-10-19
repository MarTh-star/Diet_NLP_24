from langchain_openai import ChatOpenAI
from embeddings import query_embeddings, build_chroma
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import conf
import getpass
import os
import re

query_text = "30-35 year old man, bmi of 29 trying to lose weight, using DASH diet."

columns = "Age range, Gender, Lose/Gain Weight, Diet, Pre-Condition, Foods to increase consumption of, Foods to eat in moderation, Foods to avoid, Macros: Percent of Fat, Percent of Protein, Percent of Carbs"


PROMPT_TEMPLATE = """
You are an expert on diets, give dietary adviced based only on this context:
{context}
 - -
Give dietary advice based on the above context: {question}
 - -
The advice should be only in the form of a csv delimited with a semi-colon table that can be copied into excel, and that have the following columns: {columns}
Give general as well as specific examples of the foods to increase, eat in moderation and to avoid, put each example in a different row.
Write the gender, age range, lose/gain weight, diet and percentage of macros, fat and carbs only once.
List the metadata of the context text.
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

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, columns=columns)
  
    # Initialize OpenAI chat model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Generate response text based on the prompt
    response_text = model.invoke(prompt)

    return response_text.content

if __name__ == "__main__":
    build_chroma(data_path=conf.DATA_PATH, chroma_path=conf.CHROMA_PATH, overwrite=False)
    print(query_rag(query_text))