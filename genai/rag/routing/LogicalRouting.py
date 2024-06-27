from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

from rag.query_translators import Constants


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here
        return "chain for js_docs"
    else:
        ### Logic here
        return "golang_docs"


def get_logical_route(question):
    # LLM with function call
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    # structured_llm = llm.with_structured_output(RouteQuery)
    structured_llm = Constants.LLM.with_structured_output(RouteQuery)

    # Prompt
    system = """You are an expert at routing a user question to the appropriate data source.
    
    Based on the programming language the question is referring to, route it to the relevant data source."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Define router
    router = prompt | structured_llm

    result = router.invoke({"question": question})
    print(result)
    print(result.datasource)

    full_chain = router | RunnableLambda(choose_route)
    return full_chain.invoke({"question": question})