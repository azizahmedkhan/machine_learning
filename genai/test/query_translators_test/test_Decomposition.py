from rag.query_translators import Decomposition, VectorDb

def test_generate_sub_question():
    question = "What are the main components of an LLM-powered autonomous agent system?"
    vector_store_retriever = VectorDb.load_a_blog_post_in_vector_db(
        ['https://lilianweng.github.io/posts/2023-06-23-agent/', ])
    questions = Decomposition.generate_sub_question(question)
    print(questions)
    answers = Decomposition.answers_recursively(vector_store_retriever, questions)
    print(answers)
    questions, answers = Decomposition.retrieve_and_rag(vector_store_retriever, questions)

    # Corrected indentation for the loop
    for generated_question, generated_answer in zip(questions, answers):
        print("[Question] ", generated_question)
        print("[Answer] ", generated_answer)
    final_answer = Decomposition.decomposition_answer_from_llm(question, questions, answers)
    print("[Final Answer]",final_answer)
    assert True


