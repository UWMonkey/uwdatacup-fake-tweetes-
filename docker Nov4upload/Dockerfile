FROM python:3

add Submit.py Submit.py
add model_nov4.sav model_nov4.sav
add pre_process_nov4.sav pre_process_nov4.sav

RUN pip install pandas
RUN pip install scikit-learn
RUN pip install spacy
RUN pip install joblib
RUN pip install string


RUN python -m spacy download en_core_web_sm


CMD [ "python3", "Submit.py"]