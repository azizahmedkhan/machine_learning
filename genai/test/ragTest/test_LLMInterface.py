import pytest
from rag.ragFromScratch import LLMInterface, VectorDb


class TestDataRelated():
    def test_load_a_blog_post(self):
        vector_store_retriever = VectorDb.load_a_blog_post_in_vector_db(
            ['https://lilianweng.github.io/posts/2023-06-23-agent/', ])
        print(DataRelated.ask_question(vector_store_retriever, "What is Task Decomposition?"))
        pass

    # def test_load_a_blog_post_multi_queries(self):
    #     vector_store_retriever = VectorDb.load_a_blog_post_in_vector_db(
    #         ['https://lilianweng.github.io/posts/2023-06-23-agent/', ])
    #     print(DataRelated.ask_question_multi_queries(vector_store_retriever, "What is task decomposition for LLM agents?"))
    #     pass
    #
    #
    # def test_load_a_blog_post_rag_fusion(self):
    #     vector_store_retriever = VectorDb.load_a_blog_post_in_vector_db(
    #         ['https://lilianweng.github.io/posts/2023-06-23-agent/', ])
    #     print(DataRelated.ask_question_rag_fusion(vector_store_retriever, "What is task decomposition for LLM agents?"))
    #     pass