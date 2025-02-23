FROM python:3.8

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y default-jre-headless

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY . /app
CMD streamlit run 1_🏠_Home.py --server.port=8080
