# -*- coding: utf-8 -*-

from fastapi import FastAPI
from starlette.status import HTTP_200_OK
from pydantic import BaseModel
from app import main
app = FastAPI(title='鹦鹉识别',description='识别率99.64%',version='v1.0')


class Item(BaseModel):
    image: bytes

@app.post("/birds/",status_code=HTTP_200_OK,summary='图片识别接口')
async def recognize(item : Item):
    return main(item.image)
