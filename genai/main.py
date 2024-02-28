from typing import Dict

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
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


class Item(BaseModel):
    text: str = None
    is_done: bool = False


items = []


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/items")
def create_item(item: Item):
    items.append(item)
    return items


@app.get("/items", response_model=list[Item])
def list_items(limit: int = 10):
    return items[0:limit]


@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int) -> Item:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")


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
        print(f"File contents: {contents[:100]}...")
        return DataResponse(data="File uploaded successfully")

    except Exception as e:
        print(f"Error uploading file: {e}")
        return DataResponse(data=str(e))