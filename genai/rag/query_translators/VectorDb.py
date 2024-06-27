import bs4
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import Sequence, List

from rag.query_translators import Constants


def get_vector_db() -> Chroma:
    vectorstore = Chroma(collection_name="summaries",
                         embedding_function=Constants.EMBEDDINGS)
    return vectorstore


def load_a_blog_post_in_vector_db(web_paths: Sequence[str]) -> list[Document]:
    docs = load_blog_post_from_web(web_paths)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    # Embed
    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=Constants.EMBEDDINGS)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    return retriever


def load_blog_post_from_web(web_paths: Sequence[str]) -> list[Document]:
    loader = WebBaseLoader(
        web_paths,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs
