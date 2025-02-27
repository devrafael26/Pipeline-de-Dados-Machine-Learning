### Descrição
Pipeline de Dados para análise de ações e aplicação de Machine Learning, com o objetivo de prever o preço de fechamento das ações, com atualização diária. 

A fonte de dados utilizada foi a API da Alpha Vantage, e as ferramentas do Google Cloud Platform (GCP) foram empregadas para a extração e processamento dos dados. As bibliotecas de visualização Matplotlib e Seaborn foram utilizadas dentro de um Jupyter Notebook hospedado no cluster Dataproc para análise visual dos dados.

Na etapa do Machine Learning,  por se tratar de conta gratuita,  preferi fazer no Databricks com Spark ML, por uma questão de performance. 

### Como funciona?
#### No GCP:
O Cloud Scheduler possui um job configurado para todos os dias às 20h,   acionar a cloud function que fará a requisição na API da Alpha.
Feita a requisição, os dados são armazenados no Cloud Storage em CSV.
No Dataproc, realizo o processamento dos dados e utilizo o PySpark para análises em um ambiente Jupyter hospedado no cluster.

#### Machine Learning (Databricks)

Os dados armazenados no Google Cloud Storage (GCS), são extraídos para um notebook no Databricks através da API do GCS. Utilizando o Spark ML, aplico um modelo de Regressão Linear para treinamento e previsão dos dados.
 
### Ferramentas, frameworks e bibliotecas utilizadas:

No GCP:
- Cloud Scheduler 
- Cloud Functions 
- Cloud Storage 
- Dataproc PySpark 
- Jupyter Notebook 
- Matplotlib 
- Seaborn 

No Databricks:
- PySpark 
- Spark ML 
