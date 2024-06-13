from rag.query_translators import VectorDb, StepBack, Hyde


def test_hyde_llm():
    question = "What is task decomposition for LLM agents?"
    vector_store_retriever = VectorDb.load_a_blog_post_in_vector_db(
        ['https://lilianweng.github.io/posts/2023-06-23-agent/', ])
    answer = Hyde.hyde(vector_store_retriever, question)
    print("==========================answer======================")
    print(answer)
    assert True


