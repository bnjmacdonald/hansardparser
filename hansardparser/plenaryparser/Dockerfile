# https://medium.com/bitcraft/dockerizing-a-python-3-flask-app-line-by-line-400aef1ded3a
# networking tutorial: https://docs.docker.com/network/network-tutorial-standalone/
# workon hansardparser && cd hansardparser/plenaryparser && docker build -t hansardparser/plenaryparser:latest .
# docker run -dt --rm -p 8000:8000 --name=plenaryparser --network=hansardparser --env VERBOSITY=1 hansardparser/plenaryparser
FROM ubuntu:latest
FROM python:3.6
LABEL maintainer="bnjmacdonald@gmail.com"
RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip nginx
COPY . /app
COPY config/credentials/hansardparser-1354bffbaa01.json /app/hansardparser-1354bffbaa01.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/hansardparser-1354bffbaa01.json"
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader words
# ENTRYPOINT ["python"]
# CMD ["main.py"]
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "main:app", "--timeout", "600"]
