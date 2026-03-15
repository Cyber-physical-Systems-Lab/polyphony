# Knowledge Folder for RAG

Put your grounding documents for retrieval here.

Supported file types by default:
- `.md`
- `.txt`
- `.rst`

Build the index:
```sh
python scripts/build_rag_index.py --docs_dir knowledge --db_dir rag_db
```
