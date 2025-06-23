#from pytube import youtube

from youtube_comment_downloader import *
downloader = YoutubeCommentDownloader()
import json
from youtube_comments_scraper import *
#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time as t
import os

# Import functions for data preprocessing & data preparation
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()
from sklearn import metrics
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation
import re

def comment_download(video_link):
    comments = downloader.get_comments_from_url(video_link)

     # Split the comments string into lines
    comments_lines = list(comments)

    # Convert the  comments lines into a DataFrame 
    df = pd.DataFrame(comments_lines)

    return df

def sentiment_analysis(df1):
   
   df1["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df1["Comment"]]
   df1["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df1["Comment"]]
   df1["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df1["Comment"]]
   df1['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in df1["Comment"]]
   score = df1["Compound"].values
   sentiment = []
   for i in score:
    if i >= 0.05 :
        sentiment.append('Positive')
    elif i <= -0.05 :
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
        
   df1["Sentiment"] = sentiment
   return df1

def sentiment_pie_plot(df,positive,negative,neutral):
    plt.figure(figsize=(5,6))
    return plt.show()


def main():

    st.title("YouTube comment sentiment analysis")

    # video URL
    st.write("Drop the youtube video link")
    video_link=st.text_input(" YouTube  video URL")

      # Check if the input is a valid URL and handle accordingly
    if video_link:
        st.write(f"URL entered: {video_link}")
        df=comment_download(video_link)  #df is the dataframe of the comments

        # Dropping of the unnecessary columns
        df1=df.drop(["cid","time","author","channel","votes","replies","photo","heart","reply","time_parsed"],axis=1)
        
        # renaming the column from text to comments
        df1.rename(columns={'text':"Comment"},inplace=True)

        # performing sentiment analysis
        df2=sentiment_analysis(df1)

        #df2 is the final dataframe 
        st.dataframe(df2,use_container_width=True)

        # extracting the columns for pie chart depiction
        positive=df2.loc[df2["Sentiment"]=="Positive"].count()[0]
        negative=df2.loc[df2["Sentiment"]=="Negative"].count()[0]
        neutral=df2.loc[df2["Sentiment"]=="Neutral"].count()[0]

        #pie plot
        exp=[0,0.2,0]
        labels=["Positive_comments","Negative_comments","Neutral_comments"]
        color=["#00b300","#2e2eb8","#ff6600"]
        fig=sentiment_pie_plot(df2,positive,negative,neutral)
        fig, ax = plt.subplots()
        ax.set_title("    Youtube comments Sentiments analysis   ")
        ax.pie([positive,negative,neutral],colors=color,labels=labels,explode=exp,autopct="%1.2f%%,",pctdistance=0.6,shadow=True,labeldistance=1.1)
        ax.axis('equal')
        ax.legend(loc="upper left", bbox_to_anchor=(-0.5,0))
        st.pyplot(fig,clear_figure=True,dpi=150,use_container_width=True)


    else:
        st.write("Please enter a valid YouTube URL to proceed.")
    

   
# main function
if __name__ == "__main__":
    main()







