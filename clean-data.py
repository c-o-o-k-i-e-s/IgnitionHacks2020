#Load the libraries
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
#from google.colab import files
import io
data_to_load = files.upload()
df = pd.read_csv(io.BytesIO(data_to_load['training_data.csv']))
#clean the text
def cleanText(txt):
  #remove all the @ symbols (and the usernames/text atached to them)
  txt = re.sub(r'@[A-Za-z0-9]+', '', txt) 
  #remove all the URLs
  txt = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', txt)
  #txt = re.sub(r'@[A-Za-z0-9]+', '', txt)
  #txt = re.sub(r'@[A-Za-z0-9]+', '', txt)
  #make everything lowercase
  txt = txt.lower()
  #txt = df['Text'].apply(lambda x: " ".join(word.lower() for word in x.split()))

  return txt

df['Text']=df['Text'].apply(cleanText)
print(df.Text[7])