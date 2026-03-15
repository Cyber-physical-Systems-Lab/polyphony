from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import shutil


def main() -> None:
    parser = ArgumentParser(
        description="Build a persistent RAG index from local knowledge documents",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--docs_dir", default="knowledge", type=str, help="Directory containing .md/.txt/.rst documents")
    parser.add_argument("--db_dir", default="rag_db", type=str, help="Output directory for Chroma vector DB")
    parser.add_argument("--embedding_model", default="nomic-embed-text", type=str, help="Ollama embedding model")
    parser.add_argument("--ollama_base_url", default="http://localhost:11434", type=str, help="Ollama base URL")
    parser.add_argument("--chunk_size", default=1200, type=int, help="Text chunk size")
    parser.add_argument("--chunk_overlap", default=150, type=int, help="Text chunk overlap")
    args = parser.parse_args()

    try:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import OllamaEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise SystemExit("Missing dependencies. Install with: pip install -e .[llm]") from exc

    docs_dir = Path(args.docs_dir)
    db_dir = Path(args.db_dir)
    if not docs_dir.exists():
        raise SystemExit(f"docs_dir not found: {docs_dir}")

    text_files = [p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".rst"}]
    if not text_files:
        raise SystemExit(f"No supported docs found in {docs_dir} (.md/.txt/.rst)")

    raw_docs = []
    for path in text_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        raw_docs.append(Document(page_content=text, metadata={"source": str(path)}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(200, int(args.chunk_size)),
        chunk_overlap=max(0, int(args.chunk_overlap)),
    )
    chunks = splitter.split_documents(raw_docs)
    if not chunks:
        raise SystemExit("No chunks produced from input documents")

    if db_dir.exists():
        shutil.rmtree(db_dir)

    embeddings = OllamaEmbeddings(model=args.embedding_model, base_url=args.ollama_base_url)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=str(db_dir))
    try:
        vectorstore.persist()
    except Exception:
        pass

    print(f"Indexed {len(text_files)} files into {len(chunks)} chunks.")
    print(f"RAG DB written to: {db_dir}")


if __name__ == "__main__":
    main()
