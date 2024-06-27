from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sympy import pretty_print

from rag.query_construction.tutorial_search import TutorialSearch
from rag.query_translators import Constants


def query_construction():
    system = """You are an expert at converting user questions into database queries. \
    You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
    Given a question, return a database query optimized to retrieve the most relevant results.
    
    If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm = Constants.LLM.with_structured_output(TutorialSearch)
    query_analyzer = prompt | structured_llm

    print("rag from scratch")
    print(query_analyzer.invoke({"question": "rag from scratch"}),"\n")
    query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()

    print("videos on chat langchain published in 2023")
    print(query_analyzer.invoke({"question": "videos on chat langchain published in 2023"}), "\n")
    query_analyzer.invoke(
        {"question": "videos on chat langchain published in 2023"}
    ).pretty_print()

    print("videos that are focused on the topic of chat langchain that are published before 2024")
    print(query_analyzer.invoke({"question": "videos that are focused on the topic of chat langchain that are published before 2024"}), "\n")
    query_analyzer.invoke(
        {"question": "videos that are focused on the topic of chat langchain that are published before 2024"}
    ).pretty_print()

    print("how to use multi-modal models in an agent, only videos under 5 minutes")

    analyzed = query_analyzer.invoke(
        {
            "question": "how to use multi-modal models in an agent, only videos under 5 minutes"
        })
    print(analyzed)
    analyzed.pretty_print()


from langchain_community.document_loaders import YoutubeLoader

docs = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
).load()

print(docs[0].metadata)