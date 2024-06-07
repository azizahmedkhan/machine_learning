
from langchain import hub
from langchain_openai import ChatOpenAI

PROMPT_RAG = hub.pull("rlm/rag-prompt")
LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)