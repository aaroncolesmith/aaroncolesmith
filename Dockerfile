FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app
CMD streamlit run aaroncolesmith.py --server.port 8501
