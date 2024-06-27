from rag.query_construction import query_constructor
from rag.routing import LogicalRouting, SemanticRouting


def test_get_logical_route():
    question = """Why doesn't the following code work:

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
        prompt.invoke("french")
        """

    route = LogicalRouting.get_logical_route(question)
    print("================== logical route ============== for ", question)
    print(route)
    assert True


def test_get_semantic_route():
    question = """Why doesn't the following code work:

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
        prompt.invoke("french")
        """

    route = SemanticRouting.get_query_route_symantic(question)
    print("================== semantic route ============== for ", question)
    print(route)
    question = "What's a black hole"
    route = SemanticRouting.get_query_route_symantic(question)
    print("================== semantic route ============== for ", question)
    print(route)
    assert True


def test_query_construction():
    query_constructor.query_construction()
    pass
