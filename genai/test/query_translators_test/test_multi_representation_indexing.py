from rag.indexing import multi_representation_indexing
from rag.query_translators import VectorDb
from typing import List
from langchain_core.runnables.utils import Output


def test_multi_representation_indexing():
    docs = VectorDb.load_blog_post_from_web(["https://lilianweng.github.io/posts/2023-06-23-agent/",
                                             "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"])
    summaries:List[Output] = multi_representation_indexing.summarize_each_doc(docs)
    print (summaries)
    retriever = multi_representation_indexing.store_summaries_and_docs(summaries, docs)
    retrieved_docs = multi_representation_indexing.similarity_search(retriever, "Memory in agents")
    print(retrieved_docs[0].page_content[0:500])
