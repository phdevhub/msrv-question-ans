FROM python:3.12.3

COPY * /app
WORKDIR /app

RUN pip3 install -r requeriments.txt

CMD [ "python", "app.py"]