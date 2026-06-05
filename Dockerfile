# Imagem reproduzível do QuantBot (sem o torch pesado por padrão).
# Para ligar o FinBERT: instale torch+transformers e use QUANTBOT_USE_FINBERT=1.
FROM python:3.12-slim

WORKDIR /app

# Núcleo científico que o pacote data/ importa
RUN pip install --no-cache-dir \
    numpy pandas scipy scikit-learn xgboost yfinance python-dotenv

COPY . .

# Por padrão mostra o status; troque o comando conforme o uso:
#   docker run --rm quantbot python main_income.py --weekly
#   docker run --rm quantbot python scripts/operar_paper.py
CMD ["python", "main_income.py", "--status"]
