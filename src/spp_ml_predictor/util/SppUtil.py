import pyspark.sql as ps
import pyspark.sql.functions as psf
import sys

def __fill__(pdf:ps.DataFrame
                    , window:ps.Window
                    , colname:str=None
                    , fillRangeColname:str=None
                    , fillRangeColStart=None
                    , fillRangeColEnd=None
                    , fillDirection:str='ffill') -> ps.DataFrame :
    
    pdfr:ps.DataFrame = pdf
    
    colsFilled = []
    for col in pdf.dtypes:
        colFilled = None
        if(colname and col[0] != colname): #carry forward only colname
            colFilled = psf.col(col[0])
        else: #carry forward all or only colname
            colFilled = psf.last(col[0], True).over(window) if fillDirection=='ffill' else psf.first(col[0], True).over(window)
        
        if(fillRangeColname and fillRangeColStart and fillRangeColEnd):
            colsFilled.append(psf.when((pdf[fillRangeColname] >= fillRangeColStart) & (pdf[fillRangeColname] <= fillRangeColEnd), colFilled).otherwise(psf.col(col[0])).alias(col[0]))
        else:
            colsFilled.append(colFilled.alias(col[0]))
    
    pdfr = pdfr.select(colsFilled)
            
    return pdfr

def ffill(pdf:ps.DataFrame
            , colname:str=None
            , partitionColName:str=None
            , orderingColName:str=None
            , fillRangeColname:str=None
            , fillRangeColStart=None
            , fillRangeColEnd=None) -> ps.DataFrame :
    
    pdfLocal = pdf
    if(not partitionColName):
        partitionColName = 'tempPartitioning'
        pdfLocal = pdfLocal.withColumn(partitionColName, psf.lit(partitionColName))
    
    if(not orderingColName):
        orderingColName = colname
    
    window = ps.Window.partitionBy(partitionColName).orderBy(orderingColName).rowsBetween(-sys.maxsize, 0)
    retval = __fill__(pdfLocal, window, colname, fillRangeColname, fillRangeColStart, fillRangeColEnd, 'ffill')
    
    return retval if(partitionColName != 'tempPartitioning') else retval.drop(partitionColName)
    
def bfill(pdf:ps.DataFrame
            , colname:str=None
            , partitionColName:str=None
            , orderingColName:str=None
            , fillRangeColname:str=None
            , fillRangeColStart=None
            , fillRangeColEnd=None) -> ps.DataFrame :
    
    pdfLocal = pdf
    if(not partitionColName):
        partitionColName = 'tempPartitioning'
        pdfLocal = pdfLocal.withColumn(partitionColName, psf.lit(partitionColName))
    
    if(not orderingColName):
        orderingColName = colname
    
    window = ps.Window.partitionBy(partitionColName).orderBy(orderingColName).rowsBetween(0, sys.maxsize)
    retval = __fill__(pdfLocal, window, colname, fillRangeColname, fillRangeColStart, fillRangeColEnd, 'bfill')
    
    return retval if(partitionColName != 'tempPartitioning') else retval.drop(partitionColName)
    
def fillGapsOrdered(pdf:ps.DataFrame
                                , spark:ps.SparkSession
                                , colname:str
                                , colData:list
                                , method:str='ffill') -> ps.DataFrame :
    
    extendedData = []
    for d in colData:
        t = ()
        for col in pdf.dtypes:
            t += (d,) if(col[0] == colname) else (None,)
        
        extendedData.append(t)
    
    schema = pdf.schema
    epdf = spark.createDataFrame(extendedData, schema)
    pdfr = pdf.union(epdf).dropDuplicates([colname])
    retval = pdfr
    if(method == 'ffill'):
        retval = ffill(pdfr, orderingColName=colname)
    else:
        retval = bfill(pdfr, orderingColName=colname)
    
    return retval
    
            
