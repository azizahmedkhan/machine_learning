from langchain.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from rag.query_translators import Constants


#Hypothetical Document Embeddings

def hyde(retriever: VectorStoreRetriever, question: str):
    # HyDE document genration
    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)
    generate_docs_for_retrieval = (
            prompt_hyde | Constants.LLM | StrOutputParser()
    )

    # Run
    generated_docs = generate_docs_for_retrieval.invoke({"question": question})
    print("generated_docs ===============================")
    print(generated_docs)

    # Retrieve
    retrieval_chain = generate_docs_for_retrieval | retriever
    retireved_docs = retrieval_chain.invoke({"question": question})
    print("retireved_docs ===============================")
    print(retireved_docs)

    # RAG
    template = """Answer the following question based on this context:
    
    {context}
    
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)


    return Constants.invoke_chain(prompt, {"context": retireved_docs, "question": question})

    # return final_rag_chain.invoke({"context": retireved_docs, "question": question})
