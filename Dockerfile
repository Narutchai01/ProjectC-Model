FROM python:3.9

RUN apt-get update -y

WORKDIR /code

ENV DEBIAN_FRONTEND=nointeractive

RUN pip install --upgrade pip
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get update -y
RUN apt-get install -y libglib2.0-0
# RUN apt-get install -y aptitude
# RUN aptitude install -y libglib2.0-dev

COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]