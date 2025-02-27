#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import logging
import os
from google.cloud import storage
from flask import Flask, Request
from datetime import datetime

# Usamos Flask para criarmos um servidor que irá gerenciar requisições HTTP.
app = Flask(__name__)

PROJECT_ID = "estudos-448118"
BUCKET_NAME = "dados_input"  

# Responde requisições do tipo POST
@app.route("/", methods=["POST"])  
def fechamento_gerdau(request: Request):  
    """
    Função para baixar dados e salvar no GCS.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Função iniciada")

    try:
        chave_api = os.environ.get("API_KEY")
        if not chave_api:
            logging.error("Erro: Variável de ambiente API_KEY não definida.")
            return "Erro: Variável de ambiente API_KEY não definida.", 500

        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GGBR4.SAO&apikey={chave_api}&datatype=csv'
        logging.info(f"URL da API: {url}")

        r = requests.get(url)
        r.raise_for_status()
        logging.info(f"Código de status da API: {r.status_code}")
        
        nome_arquivo = "api-acoes/fechamento_gerdau.csv"

        # Salvar o arquivo no GCS
        # blob é usado para manipular arquivos no GCS, veio através da biblioteca google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(nome_arquivo)
        blob.upload_from_string(r.text, content_type="text/csv")

        logging.info(f"Arquivo salvo com sucesso no GCS: gs://{BUCKET_NAME}/{nome_arquivo}")
        return f"Arquivo salvo no GCS: gs://{BUCKET_NAME}/{nome_arquivo}", 200

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao fazer requisição para API: {e}")
        return f"Erro ao fazer requisição para API: {e}", 500

    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
        return f"Erro inesperado: {e}", 500

