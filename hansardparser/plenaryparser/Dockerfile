# Build and run examples::
# workon hansardparser && cd hansardparser/plenaryparser && docker build -t hansardparser/plenaryparser:latest .
# docker run -dt --rm -p 8000:8000 --name=plenaryparser --network=hansardparser \
#   --env VERBOSITY=1 \
#   --env HANSARD_LINE_TYPE4_HOST="hansard_line_type4_char" \
#   --env HANSARD_LINE_HAS_SPEAKER_HOST="hansard_line_has_speaker_char" \
#   --env HANSARD_LINE_SPEAKER_SPAN_HOST="hansard_line_speaker_span_char" \
#   hansardparser/plenaryparser
FROM ubuntu:latest
FROM python:3.6
LABEL maintainer="bnjmacdonald@gmail.com"
RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip nginx
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader words
COPY . /app
COPY ./config/credentials/hansardparser-1354bffbaa01.json /app/hansardparser-1354bffbaa01.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/hansardparser-1354bffbaa01.json"
EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "main:app", "--timeout", "600"]
