FROM python:3.8

COPY ./src /api
COPY ./model /api
COPY requirements.txt /api

ENV PYTHONPATH=/api
WORKDIR /api

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]