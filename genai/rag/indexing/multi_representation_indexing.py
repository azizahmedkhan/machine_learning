import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.utils import Output
from typing import List
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from rag.query_translators import Constants, VectorDb


def summarize_each_doc(docs: List[Document]) -> List[Output]:
    chain = (
            {"doc": lambda x: x.page_content}
            | Constants.final_rag_chain(ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}"))
    )
    summaries = chain.batch(docs, {"max_concurrency": 5})
    return summaries


def store_summaries_and_docs(summaries, docs) -> MultiVectorRetriever:
    # The vectorstore to use to index the child chunks
    vectorstore = VectorDb.get_vector_db()
    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"
    # The retriever. byte_store is a doc store
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]
    # Add
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    return retriever


def similarity_search(retriever, query):
    sub_docs = retriever.vectorstore.similarity_search(query, k=1)
    print(sub_docs[0])
    retrieved_docs = retriever.get_relevant_documents(query, n_results=1)
    # return retrieved_docs[0].page_content[0:500]
    return retrieved_docs

    