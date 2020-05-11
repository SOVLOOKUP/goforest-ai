# -*- coding: utf-8 -*-
# import base64
import json
from db.query import query
from recognize.recognize import recognize




def main(photo : bytes):

    # photo type must be base64
    if type(photo) != bytes or len(photo) == 0:
        return {"status":"failed","details":"RequestTypeError"}

    # photo recognize
    result = recognize(photo)
    
    # query database
    details = query(result)

    return details


# # 本地模拟检测
# import base64
# def readima():
#     image = r'D:\Desktop\鹦鹉图谱 - 副本\山扇尾鹦鹉\山扇尾鹦鹉.jpg'
#     with open(image, 'rb') as f:
#         image = f.read()
#         return base64.b64encode(image)

# a = main(readima())
# # print(json.loads(a))
# # print(type(a))