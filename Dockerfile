FROM python:3.8-bullseye
COPY . ./
RUN pip install pystan==2.18
RUN pip install -r requirements.txt
CMD streamlit run app/aaroncolesmith.py --server.port 80




# FROM python:3.7
# COPY requirements.txt ./requirements.txt
# RUN pip install -r requiremts.txt
# ENV APP_HOME /app
# WORKDIR $APP_HOME
# COPY . ./
# CMD streamlit run app/aaroncolesmith.py --server.port $PORT


# EXPOSE 8501
# COPY ./app
# ENTRYPOINT ["streamlit", "run"]

# CMD ["app.py"]




# FROM python:3.7-slim
# COPY requirements.txt ./requirements.txt
# RUN pip install -r requirements.txt
# ENV APP_HOME /app
# WORKDIR $APP_HOME
# COPY . ./
# CMD streamlit run app/app.py --server.port $PORT