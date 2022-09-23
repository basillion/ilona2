FROM python:3.9-slim
LABEL maintainer="evgeny.tihonovich@servolux.by"
COPY web_srv.py ./app/
COPY predict.py ./app/
COPY train.py ./app/
COPY *.h5 ./app/
COPY *.pickle ./app/
COPY *.txt ./app/
COPY *.ini ./app/
COPY *.xls* ./app/
CMD ["mkdir", "/app/logvol"]
RUN pip install -r ./app/requirements.txt
RUN pip install flask
EXPOSE 5000
CMD ["python3", "/app/train.py"]
