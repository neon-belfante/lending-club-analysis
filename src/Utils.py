import pyspark
import pandas as pd
import numpy as np

from pyspark import keyword_only

from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable, MLReadable, MLWritable
from pyspark.ml import Transformer, PipelineModel
from pyspark.ml.param.shared import HasInputCols, HasOutputCols, Param, Params, TypeConverters

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import *

from sklearn.base import BaseEstimator, ClassifierMixin

spark : SparkSession = SparkSession.builder \
                    .appName('test') \
                    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.11") \
                    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
                    .config("spark.sql.execution.arrow.pyspark.enabled","true")\
                    .getOrCreate()


input_path = "../data/"
output_path = "../outputs/"


class addWoeFromSavedDF(Transformer, HasInputCols, DefaultParamsReadable, DefaultParamsWritable,
                        MLReadable, MLWritable):
    
    @keyword_only
    def __init__(self,
                 woe_df_path: str=None,
                 inputCols: list=None):
        super(addWoeFromSavedDF, self).__init__()
        kwargs = self._input_kwargs
        self.woe_df_path = Param(self, "woe_df_path", "")
        self.inputCols = Param(self, "inputCols", "")
        self.setParams(**kwargs)
        self.build_dict_flag = True
    
    @keyword_only
    def setParams(self, woe_df_path=None, inputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def setInputCols(self, value):
        """
        Sets the value of :py:attr:`inputCols`.
        """
        return self._set(inputCols=value)
    
    def buildDicts(self):
        self.woe_df = spark.read.parquet(self.getOrDefault("woe_df_path"))
        self.col_types = self.woe_df\
                        .select("column", "type")\
                        .distinct()\
                        .toPandas()\
                        .set_index("column")\
                        .to_dict()['type']
        
        self.splits = self.woe_df\
            .where(F.col("type") == "numeric")\
            .select("column", "splits")\
            .toPandas()\
            .set_index("column")\
            .to_dict()['splits']

        numeric_woe_df = self.woe_df\
            .where(F.col("type") == "numeric")\
            .select("column", "bands", "woe")\
            .toPandas()
        
        category_woe_df = self.woe_df\
            .where(F.col("type")=="category")\
            .select("column", 
                    "bands", 
                    "woe")\
            .distinct()\
            .toPandas()

        self.woe_dict = dict()
        for col_i in numeric_woe_df['column'].drop_duplicates():
            filtered_df = numeric_woe_df[numeric_woe_df['column']==col_i]
            self.woe_dict[col_i] = filtered_df.set_index("bands").to_dict()["woe"]

        self.cat_woe_dict = dict()
        for col_i in category_woe_df['column'].drop_duplicates():
            filtered_df = category_woe_df[category_woe_df['column']==col_i]
            self.cat_woe_dict[col_i] = filtered_df.set_index("bands").to_dict()["woe"]
        
        self.coalesce_woe_values_dict = self.woe_df\
            .select("column", "null_coalesce_woe")\
            .distinct()\
            .toPandas()\
            .set_index("column")\
            .to_dict()["null_coalesce_woe"]

    def _transform(self, df: DataFrame):
        if self.build_dict_flag:
            self.buildDicts()
            self.build_dict_flag = False
        result = df
        for col_i in self.getOrDefault("inputCols"):
            result = result.transform(self.addWoeColumn, col_i)
        return result

    def addWoeColumn(self, df: DataFrame, col_to_bin: str):
        column_type = self.col_types[col_to_bin]
        if column_type == "numeric":
            result = df.transform(self.addWoeNumericColumn,  col_to_bin)
        else:
            result = df.transform(self.addWoeCategoryColumn, col_to_bin)
        return result

    def addWoeNumericColumn(self, df: DataFrame, col_to_bin: str):  
        from pyspark.ml.feature import Bucketizer
        splits = self.splits[col_to_bin]
        woe_values = self.woe_dict[col_to_bin]
        coalesce_value = self.coalesce_woe_values_dict[col_to_bin]
        
        # pd.cut have a left closed interval while bucketizer have a right closed interval, this adjusts the difference
        splits = [i + 0.000000001 for i in splits]

        bucketizer = Bucketizer(inputCol=col_to_bin, outputCol=f"{col_to_bin}_bands", splits=splits)
        result = df\
                .transform(bucketizer.transform)\
                .withColumn(f"{col_to_bin}_woe", F.col(f"{col_to_bin}_bands"))\
                .replace(woe_values, subset=[f"{col_to_bin}_woe"])\
                .withColumn(f"{col_to_bin}_woe", coalesce(F.col(f"{col_to_bin}_woe"), lit(coalesce_value)))
        return result
    
    def addWoeCategoryColumn(self, df: DataFrame, col_to_bin: str):
        spark_woe = self.woe_df\
                            .where(F.col("column")==col_to_bin)\
                            .select(
                                explode("clusters").alias(col_to_bin),
                                F.col("bands").alias(f"{col_to_bin}_bands"),
                                "woe",
                                "null_coalesce_woe"
                                )
        result = df\
                .withColumn(col_to_bin, coalesce(F.col(col_to_bin), lit("None")))\
                .join(broadcast(spark_woe), [f"{col_to_bin}"], "left")\
                .withColumn(f"{col_to_bin}_woe", coalesce(F.col("woe"), F.col("null_coalesce_woe")))\
                .drop('woe', 'null_coalesce_woe')
        return result
    
    def _load(self, path):
        self.load(path)
        self.buildDicts()

class pysparkLogRegWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, pyspark_pipeline: PipelineModel):
        self.pipeline_model = pyspark_pipeline
        self.model = self.pipeline_model.stages[-1]
        self.vectorizer = self.pipeline_model.stages[0]
        self.feature_name_ = self.vectorizer.getInputCols()
        self.feature_name_in_ = self.vectorizer.getInputCols()
        self.classes_ = np.array([0,1], dtype=int)
        self.params = self.model.params
        self.intercept_ = self.model.intercept
        self.coef_ = np.array(list(self.model.coefficients), dtype=float)
        self.predictionCol = self.model.getPredictionCol()
    
    def fit(self, x, y):
        pass

    def predict_proba(self, data):
        return self.predict(data)
    
    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        result = 1/(1+np.exp(-1*((data * self.coef_).sum(axis=1) + self.intercept_)))
        # result = (data * self.coef_).sum(axis=1) + self.intercept_
        pred_2 = 1 - result
        return np.column_stack((pred_2, result))
    
    def predict2(self, data):
        if isinstance(data, DataFrame):
            df = data
        else:
            pd_df = pd.DataFrame(data).reset_index()
            df = spark.createDataFrame(pd_df, ['index'] + self.feature_name_)
        result = df\
            .transform(self.pipeline_model.transform)\
            .orderBy("index")\
            .select(collect_list(self.predictionCol))\
            .first()[0]
        result = np.array(result)
        pred_2 = 1 - result
        return np.column_stack((pred_2, result))
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags


class statsModelLogRegWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, log_reg):
        self.log_reg = log_reg
        self.feature_name_ = log_reg.params.index
        self.feature_names_in_ = log_reg.params.index
        self.classes_ = np.array([0,1], dtype=int)
        self.params = log_reg.params
        self.intercept_ = np.array(0, dtype=float)
        self.coef_ = list(self.params.values())
        self.coef_ = np.array(self.coef_, dtype=float)
    
    def fit(self, x, y):
        pass

    def predict_proba(self, data):
        pred_series = self.log_reg.predict(data)
        pred_series_2 = 1 - pred_series
        return np.column_stack((pred_series_2, pred_series))
    
    def predict(self, data):
        pred_series = self.log_reg.predict(data)
        pred_series_2 = 1 - pred_series
        return np.column_stack((pred_series_2, pred_series))
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags


from sklearn.metrics import roc_auc_score
def aucFromDf(df: pd.DataFrame, pred_col: str, target_col: str, weights: str = None):
  """
  Calculate auc from pandas dataframe
  """
  pred = df[pred_col]
  target = df[target_col]
  if weights is None:
    return roc_auc_score(target, pred, sample_weight = weights)
  else:
    return roc_auc_score(target, pred, sample_weight = df[weights])

def giniFromDf(df: pd.DataFrame, pred_col: str, target_col: str, weights: str = None):
  """
  Calculate gini from pandas dataframe
  """
  auc = aucFromDf(df, pred_col, target_col, weights)
  result = 2 * auc - 1
  return result

def aucPerGroups(df: pd.DataFrame, group_list: list, pred_col: str, target_col: str, weights:str=None):
  """
  Calculate auc by groups from pandas dataframe
  """
  result = pd.DataFrame(df.groupby(group_list).apply(aucFromDf, pred_col, target_col, weights, include_groups=True), columns=['auc']).reset_index()
  return result

def giniPerGroups(df: pd.DataFrame, group_list: list, pred_col: str, target_col: str, weights:str=None):
  """
  Calculate gini by groups from pandas dataframe
  """
  result = aucPerGroups(df, group_list, pred_col, target_col, weights)
  result['gini'] = result['auc'] * 2 - 1
  return result

def pysparkAucPerGroups(df: DataFrame, group_list: list, pred_col: str, target_col: str, weights:str = None):
  """
  Calculate auc by groups from pyspark dataframe
  """
  group_df_list = group_list + [pred_col, target_col]
  if weights is None:
    agg_column = count(lit(1))
  else:
    agg_column = sum(col(weights))
  df_grouped = df.groupBy(group_df_list).agg(agg_column.alias('count'))

  schema = ''
  for col_i in group_list:
    schema = schema + f'{col_i} {df_grouped.schema[col_i].dataType.simpleString()}, '
  schema = schema + 'auc double'
  result = df_grouped.groupBy(group_list).applyInPandas(lambda row: aucPerGroups(row, group_list, pred_col, target_col, 'count'), schema=schema)
  return result

def pysparkGiniPerGroups(df: DataFrame, group_list: list, pred_col: str, target_col: str, weights:str = None):
  """
  Calculate gini by groups from pyspark dataframe
  """
  auc_df = pysparkAucPerGroups(df, group_list, pred_col, target_col, weights)
  result = auc_df.withColumn('gini', col('auc') * 2 - 1)
  return result