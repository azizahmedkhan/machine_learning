from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

from rag.query_translators import Constants


def generate_sub_question(question: str):
    # Decomposition
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    # LLM
    llm = ChatOpenAI(temperature=0)

    # Chain
    generate_queries_decomposition = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))
    questions = generate_queries_decomposition.invoke({"question": question})
    return questions

    # Run


# question = "What are the main components of an LLM-powered autonomous agent system?"


def format_qa_pair(question, answer):
    """Format Q and A pair"""

    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


def answers_recursively(retriever, questions):
    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)

    q_a_pairs = ""
    for q in questions:
        rag_chain = (
                {"context": itemgetter("question") | retriever,
                 "question": itemgetter("question"),
                 "q_a_pairs": itemgetter("q_a_pairs")}
                | decomposition_prompt
                | Constants.LLM
                | StrOutputParser())

        answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
        return q_a_pairs


def retrieve_and_rag(retriever, sub_questions):
    """RAG on each sub-question"""

    # Initialize a list to hold RAG chain results
    rag_results = []

    for sub_question in sub_questions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)

        # Use retrieved documents and sub-question in RAG chain
        answer = (Constants.PROMPT_RAG | Constants.LLM | StrOutputParser()).invoke({"context": retrieved_docs,
                                                                                    "question": sub_question})
        rag_results.append(answer)

    return sub_questions, rag_results


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""

    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()


def decomposition_answer_from_llm(question, questions, answers):
    context = format_qa_pairs(questions, answers)

    # Prompt
    decomposition_template = """Here is a set of Q+A pairs:
    
    {context}
    
    Use these to synthesize an answer to the question: {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(decomposition_template)

    # final_amswer = final_rag_chain.invoke({"context": context, "question": question})
    # final_answer = Constants.final_rag_chain(decomposition_prompt).invoke({"context": context, "question": question})
    # return final_answer
    return Constants.invoke_chain(decomposition_prompt, {"context": context, "question": question})
