import tiktoken
from langchain_openai import OpenAIEmbeddings


def calculate_tokens_from_string(question: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(question))
    embd = OpenAIEmbeddings()
    return num_tokens


# document_result = embd.embed_query(document)
# len(query_result)

#Cosine similarity is reccomended (1 indicates identical) for OpenAI embeddings.
import numpy as np


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

#
# similarity = cosine_similarity(query_result, document_result)
# print("Cosine Similarity:", similarity)
