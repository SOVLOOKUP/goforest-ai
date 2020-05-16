# -*- coding: utf-8 -*-
# sovlookup
from fastapi import FastAPI
from starlette.status import HTTP_200_OK
from pydantic import BaseModel
from app import main
app = FastAPI(title='鹦鹉识别',description='识别率99.64%',version='v1.0')


class Item(BaseModel):
    image: bytes

@app.post("/birds/",status_code=HTTP_200_OK,summary='图片识别接口')
async def recognize(item : Item):
    if len(item.image)<40:
        return {"name":"error","socre":0,"details":"请填入正确base64格式编码的图片。"}
    #print(type(item.image))
    return main(item.image)

