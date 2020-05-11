from .bmob import Bmob


session = Bmob("f651b37a14fc05d93b35b6157ddd3112", "e351f097e2f279737ce7aded2baa6308")

def query(result):
    try:
        birdname = result.get("name","null")
        score = result.get("score",0)
    except Exception:
        pass
    response = session.find(
        "bird",
        where = {"name": birdname}, # 设置查询条件, dict或BmobQuerier
        limit = 1, # 设置最大返回行数，int
        skip = None, # 设置跳过的个数，int
        order = None, # 排序规则，str
        include = None, # 需要返回详细信息的Pointer属性，str
        keys = None, # 需要返回的属性，str
        count = None, # 统计接口: 返回数量，int
        groupby = None, # 统计接口: 根据某列分组，str
        groupcount = None, # 统计接口: 分组后组内统计数量，bool
        min = None, # 统计接口: 获取最小值，str
        max = None, # 统计接口: 获取最大值，str
        sum = None, # 统计接口: 计算总数，str
        average = None, # 统计接口: 计算平均数，str
        having = None, # 统计接口: 分组中的过滤条件，str
        objectId = None # 查询单条数据，str
    ).jsonData['results']

    # 没有查询到就返回初始结果
    if len(response) == 0:
        result['details'] = "该词条我们还没有记录，我们一定会继续努力收录哒😋"
        return result

    response = response[0]
    # add score
    response['scores'] = score
    return response



