from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Index


def loadDataSet(dataset: str):
    from datasets import load_dataset
    data = load_dataset(dataset, split="train")
    return data


def create_index(index_name: str) -> Index:
    import os
    import time
    import pinecone

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name, dimension=1536, metric="cosine"
        )
        while not pinecone.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pinecone.Index(index_name)
    index.describe_index_stats()
    return index


text_field = "text"
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")


def embed_write_text(data, index: Index):
    print("saving text in idex")
    embeds = embed_model.embed_documents([data])
    print("embeds", embeds)
    index.upsert(vectors=zip("encodedText",embeds))


from tqdm import tqdm


def embed_write_data(data, index: Index):
    print("data", data, index)
    data = data.to_pandas()

    print("Panda data", data)

    batch_size = 100

    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(i + batch_size, len(data))
        batch = data.iloc[i:i_end]
        ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()]
        texts = [x["chunk"] for _, x in batch.iterrows()]
        print(texts)
        embeds = embed_model.embed_documents(texts)
        metadata = [
            {text_field: x["chunk"],
             "title": x["title"],
             "source": x["source"]} for _, x in batch.iterrows()
        ]
        # [(id1, embed1, metadata1), (id2, embed2, metadata2), ...]

        # index.upsert(vectors=zip(ids, embeds, metadata))


def get_vector_store(index: Index) -> Pinecone:
    """

    @return:
    @param index:
    @return:
    """
    return Pinecone(
        index, embed_model, text_field
    )


def augment_prompt(query: str):
    vectorstore = get_vector_store(create_index("llama-2-rag"))
    results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""Using the contexts below, answer the query. If some information is not provided within
the contexts below, do not include, and if the query cannot be answered with the below information, say "I don't know".

Contexts:
{source_knowledge}

Query: {query}"""
    return augmented_prompt
