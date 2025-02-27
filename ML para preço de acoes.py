#!/usr/bin/env python
# coding: utf-8

# ### Machine Learning para prever o preço de fechamento da ação

# In[ ]:


import os
from pyspark.sql import SparkSession
import json
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import *


# ### OBS.:
# O motivo abaixo para copiar o arquivo é que o Spark (no contexto da Databricks) não consegue acessar diretamente os caminhos do DBFS de maneira convencional, como faria com um arquivo local no sistema de arquivos. Ao copiar o arquivo para o sistema de arquivos local, você garante que o Spark consiga acessar o arquivo corretamente.

# In[ ]:


# Copiar o arquivo do DBFS para o sistema de arquivos local
local_path = "/tmp/estudos_448118_b6a96208faf3.json"
dbutils.fs.cp("dbfs:/tmp/estudos_448118_b6a96208faf3.json", "file:" + local_path)

# Configurar credenciais no Python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_path

# Criar sessão Spark
spark = SparkSession.builder.appName("SparkML_GCS").getOrCreate()

# Configurar credenciais no Spark
spark.conf.set("fs.gs.auth.service.account.enable", "true")
spark.conf.set("google.cloud.auth.service.account.json.keyfile", local_path)

# Caminho do arquivo CSV no GCS
bucket_path = "gs://dados_input/api-acoes/fechamento_gerdau.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# Carregar o CSV no Spark
df_spark = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(bucket_path)

df_spark.show(10)


# In[ ]:


df_spark.printSchema()


# In[ ]:


df_spark = df_spark.withColumnRenamed("timestamp", "date")
df_spark.show(5)


# In[ ]:


df_spark = df_spark.withColumn("MMA-10d",round(avg("close").over(Window.orderBy(col("date").desc()).rowsBetween(-9, 0)), 2))
df_spark.show(5)


# In[ ]:


# Dividir os dados cronologicamente no Pandas
df_pandas = df_spark.toPandas()

train_size = int(len(df_pandas) * 0.2)
train_data_pandas = df_pandas[train_size:]  # 80% dos dados
display(train_data_pandas)

test_data_pandas = df_pandas[:train_size]   # 20% dos dados mais recentes
display(test_data_pandas)


# In[ ]:


# Converter de volta para Spark DataFrame
train_data_spark = spark.createDataFrame(train_data_pandas)
test_data_spark = spark.createDataFrame(test_data_pandas)


# In[ ]:


# Definir features (X) e target (y)
features = ['open', 'high', 'low', 'volume', 'MMA-10d']
assembler = VectorAssembler(inputCols=features, outputCol='features')


# In[ ]:



# Aplicar o VectorAssembler aos dados de treino e teste
train_data_spark = assembler.transform(train_data_spark)
test_data_spark = assembler.transform(test_data_spark)


# In[ ]:


# Criar e treinar o modelo (Linear Regression com Spark ML)
lr = LinearRegression(featuresCol='features', labelCol='close')
model_lr = lr.fit(train_data_spark)


# In[ ]:


# Fazer previsões
predictions_lr = model_lr.transform(test_data_spark)

# Arredondar a coluna de previsão
predictions_lr = predictions_lr.withColumn("prediction", round("prediction", 2))

# Exibir as previsões
display(predictions_lr, 10)


# ### Utilizando algumas métricas para avaliar nosso modelo

# ##### (RMSE) Root Mean Squared Error - Quanto mais próximo de zero, melhor.
# 
# - Um RMSE pequeno indica que as previsões estão bem próximas dos valores reais.
# 
# - Isso sugere que o modelo está fazendo previsões bastante precisas.

# In[ ]:


# Avaliar modelo (Spark ML)
evaluator = RegressionEvaluator(labelCol='close', predictionCol='prediction', metricName='rmse')
rmse_lr = evaluator.evaluate(predictions_lr)
print(f'RMSE: {rmse_lr:.2f}')


# - Como o preço médio de fechamento está em torno de 18-20, um erro de 0.10 representa um erro percentual muito pequeno (cerca de 0.5% do valor médio).

# ##### R² (R-quadrado), quanto mais próximo de 1, melhor!
# 
# O R² é uma métrica usada para avaliar a qualidade de um modelo de regressão. Ele indica a proporção da variabilidade dos dados que é explicada pelo modelo.
# 
# - R² = 1: O modelo explica 100% da variabilidade dos dados. Ele é perfeito.
# - R² = 0: O modelo não consegue explicar nenhuma variabilidade dos dados, ou seja, o modelo não é melhor do que simplesmente usar a média dos valores reais.
# - R² negativo: Significa que o modelo está se saindo pior do que uma simples média dos dados, o que indica que ele está fazendo previsões ruins.

# In[ ]:


r2_evaluator = RegressionEvaluator(labelCol='close', predictionCol='prediction', metricName='r2')
r2 = r2_evaluator.evaluate(predictions_lr)
print(f'R²: {r2:.2f}') 


# ##### Erro percentual 
# - Calcula a relaçao da predição com relação ao valor real de fechamento.
# - Quanto mais baixo, melhor.

# In[ ]:


# Calcular erro percentual para cada linha
predictions_lr = predictions_lr.withColumn('erro_percentual', round(abs(col('prediction') - col('close')) / col('close') * 100, 2))
predictions_lr.select('date', 'close', 'prediction', 'erro_percentual').show(10)

# Calcular a média do erro percentual
avg_erro_percentual = predictions_lr.agg({'erro_percentual': 'avg'}).collect()[0][0]
print(f'Média do Erro Percentual: {avg_erro_percentual:.2f}%')


# ## Considerações finais sobre o modelo
# 
# As métricas RMSE (0,10), R² (0,96) e o erro percentual (0,43%) indicam que o modelo apresenta um bom desempenho na previsão do preço de fechamento (close). O valor de R² sugere uma forte correlação entre as previsões e os valores reais, enquanto o RMSE e o erro percentual indicam que as previsões estão próximas dos valores reais, com um pequeno desvio.
# 
# É recomendável realizar testes com mais variáveis (features), como as máximas e mínimas semanais, além de outros indicadores técnicos. Esses testes podem fornecer uma visão mais completa do comportamento do modelo e possibilitar melhorias nas métricas de avaliação.
# 
# Esses ajustes ajudam a tornar mais claro o impacto das métricas e reforçam a ideia de que adicionar mais variáveis pode melhorar o modelo.
