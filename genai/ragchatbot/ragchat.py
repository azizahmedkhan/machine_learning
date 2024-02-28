# from langchain.chat_models import ChatOpenAI
# import os
# import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# openai.api_key = os.environ["OPENAI_API_KEY"]

# from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model_name="gpt-3.5-turbo")

# from langchain.schema import (
#     SystemMessage,
#     HumanMessage,
#     AIMessage
# )

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to understand string theory.")
]

# # res = chat(messages)
# # print(res.content)
# # messages.append(res)
# prompt = HumanMessage(content="Why do physicists believe it can produce a 'unified theory'?")

# messages.append(prompt)
# # res = chat(messages)
# # print(res.content)

# print(len(messages))
# # messages.append(res)
# # print(len(messages))

# # prompt = HumanMessage(content="What is so special about Llama 2?")
# # messages.append(prompt)

# # res = chat(messages)
# # print(res.content)

# # messages.append(res)
# # prompt = HumanMessage(content="Can you tell me about the LLMChain in LangChain?")
# # messages.append(prompt)

# # res = chat(messages)
# # print(res.content)

# # # A description of LLMChains, Chains, and LangChain 
# llmchain_information = [
#     "A LLMChain is the most common type of chain. It consists of a PromptTemplate, a model (either an LLM or a ChatModel), and an optional output parser. This chain takes multiple input variables, uses the PromptTemplate to format them into a prompt. It then passes that to the model. Finally, it uses the OutputParser (if provided) to parse the output of the LLM into a final format.",
#     "Chains is an incredibly generic concept which returns to a sequence of modular components (or other chains) combined in a particular way to accomplish a common use case.",
#     "LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model via an api, but will also: (1) Be data-aware: connect a language model to other sources of data, (2) Be agentic: Allow a language model to interact with its environment. As such, the LangChain framework is designed with the objective in mind to enable those types of applications."
# ]
# len(llmchain_information)
# source_knowledge = "\n".join(llmchain_information)

# query = "Can you tell me about the LLMChain in LangChain?"

# augmented_prompt = f"""Using the contexts below, answer the query. If some information is not provided within
# the contexts below, do not include, and if the query cannot be answered with the below information, say "I don't know".

# Contexts:
# {source_knowledge}

# Query: {query}"""
# # print(augmented_prompt)

# # print(messages[-1])


# messages[-1] = HumanMessage(content=augmented_prompt)

# # print(messages[-1])

import utils
data = utils.loadDataSet("jamescalam/llama-2-arxiv-papers-chunked")
# index = utils.create_index("llama-2-rag")
# print ('index created')
# utils.embed_write_data(utils.loadDataSet("jamescalam/llama-2-arxiv-papers-chunked") , index)
# vectorstore = utils.get_vector_store(index)

query = "What is so special about Llama 2?"

# print(vectorstore.similarity_search(query, k=3))

prompt = HumanMessage(content=utils.augment_prompt(query))

messages.append(prompt)
res = chat([HumanMessage(content=utils.augment_prompt("What safety measures were used in the development of llama 2?"))])
print(res.content)