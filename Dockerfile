FROM python:3.10.10
COPY . /app
WORKDIR /app

RUN apt update -y && apt install awscli -y
RUN apt-get update && pip install -r requirements.txt

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app