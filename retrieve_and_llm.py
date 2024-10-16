from langchain_openai import ChatOpenAI
from embeddings import query_embeddings
import conf
import getpass
import os

query_text = "I am a 30 year old woman with a bmi of 21.9, I have no dietary preferences and I am trying to lose weight."
columns = ["Foods to increase consumption of", "Foods to eat in moderation", "Foods to avoid", "(For intermittent fasting) Eating time window", "Duration (numeric - in weeks) (for intermittent fasting)", "Macros: Percent of Fat, Percent of Protein, Percent of Carbs"]

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
 - -
The answer should be in the form of a table that have the following columns:{columns}
"""

def query_rag(query_text):
    retriever = Chroma(persist_directory=str(conf.CHROMA_PATH))
    results = query_embeddings(conf.CHROMA_PATH, query_text)

    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
  
    # Initialize OpenAI chat model
    model = ChatOpenAI(model="gpt-4o-mini")

    # Generate response text based on the prompt
    response_text = model.predict(prompt)

    return response_text

if __name__ == "__main__":
    print(query_rag(query_text))