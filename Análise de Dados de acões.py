#!/usr/bin/env python
# coding: utf-8

# ## Análise de Dados de acões com PySpark, Matplotlib e Seaborn no GCP

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pandas as pd
from io import StringIO
import time


# In[2]:


##### Criando a sessão Spark
spark = SparkSession.builder.appName("ProcessamentoCSV").getOrCreate()

##### Lendo o arquivo CSV 
df_spark = spark.read.csv("gs://dados_input/api-acoes/fechamento_gerdau.csv", header=True, inferSchema=True)
df_spark.show()


# In[3]:


df_spark.printSchema()


# In[4]:


df_spark = df_spark.withColumnRenamed("timestamp", "date")
df_spark.show()


# In[5]:


# Analisando alguns dados de nosso df.

menor_preco = df_spark.orderBy(col("low").asc()).select("date", "low").first()

maior_preco = df_spark.orderBy(col("high").desc()).select("date", "high").first()

min_vol = df_spark.orderBy(col("volume").asc()).select("date", "volume" ).first()

max_vol = df_spark.orderBy(col("volume").desc()).select("date", "volume" ).first()

media_fechamento = df_spark.agg(round(avg("close"), 2).alias("media_close")).collect()

# Exibir os resultados
print(f"Menor preço: {menor_preco['low']} em {menor_preco['date']}")
print(f"Maior preço: {maior_preco['high']} em {maior_preco['date']}")
print(f"Menor volume: {min_vol['volume']} em {min_vol['date']}")
print(f"Maior volume: {max_vol['volume']} em {max_vol['date']}")
print("Média de fechamento dos últimos 100 dias:", media_fechamento[0]["media_close"])


# ### Vamos calcular a média móvel simples (SMA) dos últimos 10 dias, para tentarmos identificar alguma tendência.

# In[6]:


from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Window.orderBy("date"): Organiza os dados pela data.
# rowsBetween(-9, 0): Define uma janela de 10 dias, incluindo o dia atual (índice 0) e os 9 dias anteriores. 

df_spark = df_spark.withColumn("SMA_10",F.round(F.avg("close").over(Window.orderBy(F.col("date").desc()).rowsBetween(-9, 0)), 2)
)
df_spark.show()


# <font color="red"><h4>Comentário a respeito do aviso WARN WindowExec:
# <font color="red"><h5>Ao usar a operação de janela ‘Window’ sem especificar uma partição, faz com que o PySpark mova todos os dados para uma única partição, algo que pode prejudicar o desempenho. Como nossa base de dados é pequena, vou ignorar esse aviso.

# ### Importando as bibliotecas Matplotlib e Seaborn para construção dos gráficos

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Converte o DataFrame Spark para Pandas
df_pandas = df_spark.select("date", "SMA_10").toPandas()


# In[10]:


# Aqui é importante converter para Pandas e garantir que a coluna date seja datetime

df_plot = df_spark.toPandas()
df_plot["date"] = pd.to_datetime(df_plot["date"], format="%d-%m-%Y")

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_plot, x="date", y="SMA_10", marker="o", linestyle="-", color="b", label="SMA 10")

# Ajustes visuais
plt.xlabel("Data")
plt.ylabel("SMA 10")
plt.title("Média Móvel de 10 Períodos (SMA 10)")
plt.xticks(rotation=45)
plt.grid()

plt.show()


# ### Gráfico Preço de fechamento x Volume (em milhões)

# In[11]:


df_pandas2 = df_spark.select("date", "close", "volume").toPandas()

df_pandas2["date"] = pd.to_datetime(df_pandas2["date"], format="%d-%m-%Y")

# Ajustar escala do volume (milhões)
df_pandas2["volume_milhoes"] = df_pandas2["volume"] / 1_000_000

# Criar figura e eixo
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotar o preço de fechamento (linha)
ax1.plot(df_pandas2["date"], df_pandas2["close"], color="blue", label="Preço de Fechamento", linewidth=2)
ax1.set_xlabel("Data")
ax1.set_ylabel("Preço de Fechamento", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Criar segundo eixo para volume (barras)
ax2 = ax1.twinx()
ax2.bar(df_pandas2["date"], df_pandas2["volume_milhoes"], color="orange", alpha=0.3, label="Volume (Milhões)")
ax2.set_ylabel("Volume (Milhões)", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

# Adicionar legendas
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# layout
plt.title("Preço de Fechamento x Volume (em Milhões)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)

plt.show()


# ### Análise de Correlação entre Preço e Volume

# In[13]:


df_spark.select(round(F.corr("volume", "close"), 2).alias("corr_volume_close")).show()


# O coeficiente de correlação de 0.17 entre o volume de negociações e o preço de fechamento (close) indica uma correlação fraca e positiva.
# 
# ### Interpretação:
# * Próximo de 1: Forte correlação positiva (quando o volume aumenta, o preço tende a subir).
# * Próximo de -1: Forte correlação negativa (quando o volume aumenta, o preço tende a cair).
# * Próximo de 0: Pouca ou nenhuma correlação linear entre volume e preço.

# ### Comparação entre Variação de Preço e Volume

# #### Cálculo da Variação de Preço (diff_price):
# 
# * Objetivo: Verificar se grandes variações no preço são acompanhadas por aumento no volume de transações.
#     
# * Como fazer: Calcule a diferença percentual do preço e a diferença no volume e compare.
#     
# * diff_price: Mostra a variação percentual de preço de fechamento em relação ao preço de abertura no mesmo dia.
# 
# #### Cálculo da Variação de Volume (diff_volume):
# 
# * Objetivo: Calcular a variação percentual do volume de transações entre o dia atual e o dia anterior. A função lag é usada para pegar o valor do volume do dia anterior.
#     
# * diff_volume: Mostra a variação percentual do volume de transações de um dia para o próximo, comparando com o volume do dia anterior.
# 

# In[15]:


df_spark = df_spark.withColumn(
    "diff_price(%)", F.round((df_spark["close"] - df_spark["open"]) / df_spark["open"] * 100, 2)
)

df_spark = df_spark.withColumn(
    "diff_volume(%)", F.round(
        (df_spark["volume"] - F.lag(df_spark["volume"]).over(Window.orderBy(F.col("date")))) /
        F.lag(df_spark["volume"]).over(Window.orderBy(F.col("date"))) * 100, 2
    )
)

df_spark = df_spark.orderBy(F.col("date").desc())
df_spark.show()


# ### Considerações sobre diff_price(%) e diff_volume(%)
# * No dia 31-01-2025 ocorreu uma variação negativa do preço, e se observarmos a variação do volume com relação ao dia anterior, houve um aumento de 117%.
# 
# * Seguindo, no dia 03-02-2025, ocorre também uma variação relevante do preço, porém positiva, e para surpresa, com -28% de volume comparando com o dia anterior. 
# 
# * No dia 10-02-2025, temos mais uma vez uma boa variação do preço positiva, acompanhado de 156% de volume a mais, quando comparado com o dia anterior.
# 
# * No dia 20-02-2025 tivemos uma forte variação do preço, acompanhado de um relevante aumento do volume de 200% com relação ao dia anterior.
# 
# ##### Podemos concluir que, na maioria das vezes que o preço varia acima de 2%, seja positivamente ou negativamente, a variação de volume com relação ao dia anterior, aumenta, sendo que tem dias que o volume é consideravelmente maior do que no dia anterior.
# 
