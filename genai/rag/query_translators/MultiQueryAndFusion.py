from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.load import dumps, loads

from rag.query_translators import Constants


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(retriever: VectorStoreRetriever, chat_question: str):
    # Prompt
    # template = """Answer the question based only on the following context:
    # {context}

    # Question: {question}
    # """

    # prompt = ChatPromptTemplate.from_template(template)
    # print(prompt)
    # print(calculate_tokens_from_string(chat_question, "cl100k_base"))

    # Chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | Constants.final_rag_chain(Constants.PROMPT_RAG)
    )
    # Question
    return rag_chain.invoke(chat_question)


def generate_multi_queries(question: str):
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    return query_generation_chain(prompt_perspectives)


def rag_fusion_queries(question: str):
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    return query_generation_chain(prompt_rag_fusion)


def query_generation_chain(chatPromptTemplate: ChatPromptTemplate):
    generated_queries = (
            Constants.final_rag_chain(chatPromptTemplate)
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
    )
    return generated_queries


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


# question = "What is task decomposition for LLM agents?"
# retrieval_chain = generate_multi_queries | retriever.map() | get_unique_union
# docs = retrieval_chain.invoke({"question": question})
# len(docs)

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def ask_question_from_llm(retrieval_chain, question: str):
    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    print("retrieval_chain >>", retrieval_chain)
    print("==================================\n\n")
    final_rag_chain = (
            {"context": retrieval_chain,
             "question": itemgetter("question")}
            | Constants.final_rag_chain(prompt)
    )
    answer = final_rag_chain.invoke({"question": question})
    return answer


def ask_question_multi_queries(retriever: VectorStoreRetriever, question: str):
    retrieval_chain = generate_multi_queries | retriever.map() | get_unique_union
    return ask_question_from_llm(retrieval_chain, question)


def ask_question_rag_fusion(retriever: VectorStoreRetriever, question: str):
    retrieval_chain_rag_fusion = rag_fusion_queries | retriever.map() | reciprocal_rank_fusion
    return ask_question_from_llm(retrieval_chain_rag_fusion, question)
