#Load the libraries
import pandas as pd
import re
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

#make a progress bar so we don't cry thinking nothing's happening
tqdm.pandas()

#print an update
print("Getting data")

#load the csv file as a dataframe
df = pd.read_csv('contestant_judgment.csv')

#calculate the number of rows in the dataframe
df_len = len(df.index)
#display the number of rows
print(f"Data retrieved. Length: {df_len}")

#function to clean the text
def cleanText(original_text):
  #remove all the @ symbols (and the usernames/text atached to them)
  original_text = re.sub(r'@[A-Za-z0-9]+', '', original_text) 
  #remove all the URLs
  original_text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', original_text)
  #remove punctuation
  original_text = re.sub(r'[^\w\s]', '', original_text) 
  #make everything lowercase
  original_text = original_text.lower()
  #return the cleaned text
  return original_text

#apply the cleaned text to the dataframe
df['Text']=df['Text'].apply(cleanText)

#Make a variable so the code looks cleaner
SIA = SentimentIntensityAnalyzer()
#function to get the sentiment
def sentiment_analysis(cleaned_text):
  #store the sentiment scores (which calculates how positive, negative, and neutral the text is)
  sentiment_score = SIA.polarity_scores(cleaned_text)
  #store the positive and negative values of the text
  negative_score = sentiment_score['neg']
  positive_score = sentiment_score['pos']
  #print 1 if the sentence is more positive than negative or it it's neutral
  if positive_score >= negative_score:
    return "1"
  #print 0 if the sentence has a negative sentiment
  else:
    return "0"

#display an update
print("Analyzing Data...")

#create a column to store the sentiment score in
df['Sentiment']=df['Text'].progress_apply(sentiment_analysis)

#display yet another update becuz we're kool like that *insert sunglasses emoji here*
print("Analysis Complete!")

#print this to a csv file
df.to_csv("results.csv", index=False)

#idk if you actually look at the code (or the comments for that matter) but thank you so much for hosting this hackathon! It was tons of fun and we learned so much so thank you =D