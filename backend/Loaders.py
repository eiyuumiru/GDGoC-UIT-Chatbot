from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

def load_markdown(data_dir: str = "backend/dataset"):
    p = Path(data_dir)
    loader = DirectoryLoader(
        str(p),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={"mode": "single"}
    )
    return loader.load()
