from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PROMPT_RAG = hub.pull("rlm/rag-prompt")
LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
EMBEDDINGS = OpenAIEmbeddings()

def final_rag_chain(prompt):
    return (
            prompt
            | LLM
            | StrOutputParser()
    )


def invoke_chain(prompt, dict):
    return (
            prompt
            | LLM
            | StrOutputParser()
    ).invoke(dict)
