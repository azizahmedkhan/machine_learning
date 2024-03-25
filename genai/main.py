from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ragchatbot import utils
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

app = FastAPI()
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8000)
origins = [
    "http://localhost:3000",  # Example: Allow requests from localhost during development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
chat = ChatOpenAI(model_name="gpt-3.5-turbo")
messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi AI, how are you today?"),
        AIMessage(content="I'm great thank you. How can I help you?"),
        HumanMessage(content="I'd like to understand string theory.")
    ]


@app.get("/")
def root():
    return {"Hello": "World"}


class FileInfo(BaseModel):
    filename: str
    content_type: str


# allowed_content_types = ["image/jpeg", "image/png", "application/pdf"]
#
#
# def validate_content_type(file: File):
#     if file.content_type not in allowed_content_types:
#         raise ValueError("Unsupported content type")
#
#
# def upload_file_validator(func):
#     async def wrapper(file: File, **kwargs):
#         # Content-type check
#         validate_content_type(file)
#         # ... Additional validation logic here ...
#         return await func(file, **kwargs)
#
#     return wrapper

class DataResponse(BaseModel):
    data: str


@app.post("/upload-file/")
async def upload_file(file: UploadFile) -> DataResponse:
    try:
        contents = await file.read()
        contents.decode('utf-8')
        print(f"File contents: {contents[:100]}...")

        index = utils.create_index("llama-2-rag")
        print('index created')
        # utils.embed_write_data(utils.loadDataSet("jamescalam/llama-2-arxiv-papers-chunked") , index)
        utils.embed_write_text(contents.decode('utf-8'), index)
        # vectorstore = utils.get_vector_store(index)

        # chat_message("What is so special about Llama 2?")
        return DataResponse(data="File uploaded successfully")

    except Exception as e:
        print(f"Error uploading file: {e}")
        return DataResponse(data=str(e))


class ChatMessage(BaseModel):
    message: str


@app.post('/chat-message')
def chat_message(chat_data: dict):
    print(chat_data)
    try:
        message = chat_data.get('message')
        print(message)
        # message = chat_message.message
        # openai.api_key = os.environ["OPENAI_API_KEY"]

        # print(vectorstore.similarity_search(query, k=3))

        # prompt = HumanMessage(content=utils.augment_prompt(message))

        # messages.append(prompt)
        res = chat(
            [HumanMessage(content=utils.augment_prompt(message))])
        print(res.content)
        return {'status': 'success', 'message': res.content}
    except Exception as e:
        return {'status': 'error', 'error_message': str(e)}


@app.post("/chat-message1")
def chat_message1(message: str) -> str:
    print("chat_message", message)
    # openai.api_key = os.environ["OPENAI_API_KEY"]

    # print(vectorstore.similarity_search(query, k=3))

    prompt = HumanMessage(content=utils.augment_prompt(message))

    messages.append(prompt)
    res = chat(
        [HumanMessage(content=utils.augment_prompt("What is covered in Natural Language Generation?"))])
    print(res.content)
    return res.content
