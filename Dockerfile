FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY requirements_torch.txt .
RUN pip install -r requirements_torch.txt

COPY out/model_attention_dataset-to-tensor_30017_10_0.0002_0.2.pth ./model.pth
COPY models.py .
COPY api_prediction.py .

EXPOSE 5000

CMD ["python", "api_prediction.py", "--device=cpu", "--model_path=model.pth"]
