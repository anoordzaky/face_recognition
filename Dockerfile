FROM python:3.7.16-bullseye

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY repo_requirements.txt .
RUN pip install -r repo_requirements.txt

RUN mkdir /app
WORKDIR /app

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]