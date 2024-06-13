from rag.query_translators import VectorDb, StepBack


def test_generate_sub_question():
    question = "What is task decomposition for LLM agents?"
    vector_store_retriever = VectorDb.load_a_blog_post_in_vector_db(
        ['https://lilianweng.github.io/posts/2023-06-23-agent/', ])
    step_back_question = StepBack.generate_step_back_queries_from_llm(vector_store_retriever, question)
    answer = StepBack.step_back_answer_from_llm(vector_store_retriever, question, step_back_question)
    print(answer)
    assert True


