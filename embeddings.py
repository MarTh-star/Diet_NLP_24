import shutil
from pathlib import Path
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
import conf


def load_documents(data_path: Path) -> list[Document]:
    """Parse dietary data from the csv format to langchain.Document for all files under data_path."""
    docs = []
    for diet_type_path in data_path.iterdir():
        for chunk_path in diet_type_path.iterdir():
            df = pd.read_csv(chunk_path, index_col=0)
            df.dropna(
                subset="chunk_text", inplace=True
            )  # drop rows with missing chunk_text

            # Skip documents which only contained empty rows
            if len(df) == 0:
                print(f"Skipping empty document {chunk_path}")
                continue

            df["diet_type"] = diet_type_path.name
            df["chunk_id"] = f"{diet_type_path.name}_{chunk_path.name}"

            for _, row in df.iterrows():
                doc = Document(
                    page_content=row["chunk_text"],
                    metadata={"source": f"{row['chunk_id']}_{row['chunk_label']}"},
                )
                docs.append(doc)

    return docs


def save_to_chroma(
    documents: list[Document], chroma_path: Path, batch_size: int = 40000
) -> None:
    """Saves list of documents into ChromaDB under specified path."""
    embeddings = OpenAIEmbeddings(api_key=conf.OPENAI_API_KEY)
    total_docs = len(documents)
    start = load_last_processed_index(chroma_path)

    while start < total_docs:
        end = min(start + batch_size, total_docs)
        batch_docs = documents[start:end]
        Chroma.from_documents(
            batch_docs,
            embeddings,
            persist_directory=str(chroma_path),
        )
        print(f"Saved {len(batch_docs)} chunks to {chroma_path}.")
        update_last_processed_index(chroma_path, end)
        start += batch_size


def load_last_processed_index(chroma_path: Path) -> int:
    """Loads index of the last processed chunk."""
    progress_file = chroma_path / conf.PROGRESS_LOG
    if progress_file.exists():
        with open(progress_file, "r") as file:
            last_index = int(file.read().strip())
            return last_index
    return 0


def update_last_processed_index(chroma_path: Path, last_index: int) -> None:
    """Update index of the last processed chunk by saving it to the file."""
    progress_file = chroma_path / conf.PROGRESS_LOG
    with open(progress_file, "w") as file:
        file.write(str(last_index))


def build_chroma(data_path: Path, chroma_path: Path, overwrite: bool = False) -> None:
    """Construct full ChromaDB which can be used later to query embeddings.
    
    If the ChromaDB already exists, it won't be overwritten, unless explicitely used with the overwrite param.
    """
    if overwrite and chroma_path.exists():
        shutil.rmtree(chroma_path)

    if not chroma_path.exists():
        documents = load_documents(data_path)
        save_to_chroma(documents, chroma_path)


def query_embeddings(
    chroma_path: Path, query: str, top_k: int = 50, threshold: float = 0.9
) -> list[Document]:
    """Get embeddings for the specified query.
    
    Number of embeddings returned is parametrized by the top_k param.
    """
    db = Chroma(
        persist_directory=str(chroma_path), embedding_function=OpenAIEmbeddings()
    )

    # Retrieving the context from the DB using similarity search
    relevant_documents = db.similarity_search_with_relevance_scores(query, k=top_k)
    return relevant_documents
    # TODO: Filter out documents with scores below the threshold
    # return [doc for doc, score in relevant_documents if score >= threshold]


if __name__ == "__main__":
    build_chroma(
        data_path=conf.DATA_PATH, chroma_path=conf.CHROMA_PATH, overwrite=False
    )
    relevant_documents = query_embeddings(
        conf.CHROMA_PATH, "What is actually ketogenic diet and how it can help me?"
    )
    for doc in relevant_documents:
        print(doc.page_content)
        print("=" * 80)
