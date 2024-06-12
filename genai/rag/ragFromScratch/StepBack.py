from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever

from rag.ragFromScratch import Constants

def generate_step_back_queries_from_llm(retriever: VectorStoreRetriever, question: str):
    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?",
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?",
        },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )
    generate_queries_step_back = prompt | Constants.LLM | StrOutputParser()
    # question = "What is task decomposition for LLM agents?"
    step_back_queries = generate_queries_step_back.invoke({"question": question})
    return step_back_queries


def step_back_answer_from_llm(retriever: VectorStoreRetriever, question: str, step_back_query):
    # Response prompt
    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.
    
    # {normal_context}
    # {step_back_context}
    
    # Original Question: {question}
    # Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

    chain = (
            {
                # Retrieve context using the normal question
                "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
                # Retrieve context using the step-back question
                # "step_back_context": step_back_queries | retriever,
                "step_back_context": lambda x: x["step_back_query"],
                # Pass on the question
                "question": lambda x: x["question"],
            }
            | response_prompt
            | Constants.LLM
            | StrOutputParser()
    )

    answer = chain.invoke({"question": question,"step_back_query":step_back_query})
    return answer
