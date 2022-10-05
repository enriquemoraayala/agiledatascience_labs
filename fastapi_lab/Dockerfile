#Dockerfile
FROM python:3.8
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
COPY ./app.py .
COPY ./model ./model
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]