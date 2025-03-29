# src/config.py
import os
import yaml

with open("openai_key.yaml", 'r') as file:
    api_creds = yaml.safe_load(file)
os.environ["OPENAI_API_KEY"] = api_creds['openai_key']