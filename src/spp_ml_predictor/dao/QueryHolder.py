
queryMap = {}


def __getQueryTag__(queryFilePath, tag):
    return (queryFilePath + "[" + tag + "]")

def __putQuery__(queryFilePath, tag, query):
    queryTag = __getQueryTag__(queryFilePath, tag)
    if queryTag in queryMap:
        raise KeyError('Query with filepath {} and tag {} mentioned twice. Please check either file or loading mechanism'.format(queryFilePath, tag))

    queryMap[queryTag] = query

def readQueries1(queryFilePath, queryIdPrefix, queryIdPostfix):
    qf = open(queryFilePath, 'r')
    queryId = None
    query = None
    line = None

    for line in qf:
        if(line.startswith(queryIdPrefix)):
            if(queryId):
                __putQuery__(queryFilePath, queryId, query)

            queryId = line.strip().removeprefix(queryIdPrefix).removesuffix(queryIdPostfix)
            query = ""
        elif(not line == ""):
            query = query + line

    if (queryId):
        __putQuery__(queryFilePath, queryId, query)

def readQueries2(queryFilePath, queryIdPrefix, queryIdPostfix):
    readQueries1(queryFilePath, queryIdPrefix, queryIdPostfix, None)

def getQuery(queryFilePath, tag):
    queryTag = __getQueryTag__(queryFilePath, tag)
    if queryTag not in queryMap:
        raise KeyError('Query with filepath {} and tag {} not found. Please check either file or loading mechanism'.format(queryFilePath, tag))

    return queryMap.get(queryTag)