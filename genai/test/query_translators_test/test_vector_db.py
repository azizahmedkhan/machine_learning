from rag.query_translators import VectorDb


def test_document_load_in_vector_db():
    print("test_document_load in vector db")
    print(VectorDb.load_a_blog_post_in_vector_db(['https://lilianweng.github.io/posts/2023-06-23-agent/', ]))
    pass


def test_document_load():
    print("test_document_load")
    docs = VectorDb.load_blog_post_from_web(["https://lilianweng.github.io/posts/2023-06-23-agent/",
                                             "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/"])
    print(docs)
    pass
