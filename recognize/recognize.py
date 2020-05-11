# import requests
import json
from collections import Counter
from .bird_m import predict

def my_model(photo) -> dict:
    
    try:
          return predict.main(photo)
    except Exception as e:
          return {'name':str(e),'score':0}


# def baidu(photo) -> dict:
#     headers = {
#         "Content-Type":"application/x-www-form-urlencoded"
#     }
#     data = {
#         "image": photo,
#         "top_num": 1,
#         "baike_num": 0
#     }
#     url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal?access_token=24.b8c7c15554c89181584f5999d4e74ff3.2592000.1589974368.282335-19520935"

#     response = requests.post(url=url,headers=headers,data=data).content

#     try:
#         return json.loads(response)['result'][0]
#     except Exception:
#         return {'name':'baiduerror','score':0}

# def ali1(photo) -> dict:
#     headers = {
#         "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
#         "Authorization":"APPCODE 9dabb747136f4cd295929474e3a75765"
#     }
#     data = {
#         "image": b"data:image/jpg;base64," + photo
#     }
#     url = "https://ocranimals.market.alicloudapi.com/animal"

#     response = requests.post(url=url,headers=headers,data=data).content
#     # print(response)
#     try:
#         return {"name":json.loads(response)['animalType'],"score":0.5}
#     except Exception:
#         return {'name':'ali1error','score':0}

# def ali3(photo) -> dict:
#     headers = {
#         "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
#         "Authorization":"APPCODE 9dabb747136f4cd295929474e3a75765"
#     }
#     data = {
#         "image": photo,
#         "num": 1,
#         "baike": 0
#     }
#     url = "https://dongwu.market.alicloudapi.com/do"

#     response = requests.post(url=url,headers=headers,data=data).content

#     try:
#         return json.loads(response)['result'][0]
#     except Exception:
#         return {'name':'ali3error','score':0}

def recognize(photo : bytes) -> dict:

    # 添加算法结果
    results = [my_model(photo)]

    #print(results)
    
    # 一个算法判断就直接返回结果
    if len(results) == 1:
        return results[0]

    # 多个算法判断
    # 投票出最终结果
    count_result = Counter(map(lambda x : x.get('name','null'),results)).most_common(1)[0]
    # 投票无法解决就获取概率大的
    if count_result[1] == 1:
        final_result = max(results,key=lambda x : float(x.get('score',0)))
    else:
        for result in results:
            if result.get('name','null') == count_result[0]:
                final_result = result
                break
    #print(final_result)
    return final_result
    



# 获得百度应用access_token
# import requests
# import json
# data = {
#     "grant_type":"client_credentials",
#     "client_id":"ZBTXzZVP1o2INVDGtyG8Eayi", 
#     "client_secret":"331hvDGWVnULIUUL5ImXMzshDa12ZMsN"
#     }
# a = requests.post("https://aip.baidubce.com/oauth/2.0/token",data)

# print(json.loads(a.content))
