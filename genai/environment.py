import os


def get_pinecone_key():
    return os.environ["PINECONE_API_KEY"]


def get_pinecone_environment():
    return os.environ["PINECONE_ENVIRONMENT"]


def get_open_api_key():
    return os.environ["OPEN_AI_Key"]


def get_hugging_face_key():
    return os.environ["HUGGING_FACE_ACCESS_TOKEN"]


def get_voyage_key():
    return os.environ["VOYAGE_API_KEY"]


def get_anthropic_key():
    return os.environ["ANTHROPIC_KEY"]


def get_lang_chain_tracing():
    return os.environ["LANGCHAIN_TRACING_V2"] | "true"


def get_lang_chain_endpoint():
    return os.environ["LANGCHAIN_ENDPOINT"] | "https://api.smith.langchain.com"


def get_lang_chain_key():
    return os.environ["LANGCHAIN_API_KEY"]


def get_cohere_key():
    return os.environ["COHERE_API_KEY"]
