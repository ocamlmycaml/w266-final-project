FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter

COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /nltk_data
ENV NLTK_DATA /nltk_data
RUN python -c "import nltk; nltk.download('punkt')"

