FROM python:3.8

EXPOSE 8501

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

COPY . /app

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]