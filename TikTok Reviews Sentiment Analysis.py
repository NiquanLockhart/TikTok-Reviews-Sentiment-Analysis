#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#TikTok Reviews Sentiment Analysis using Python


# In[1]:


pip install pandas


# In[2]:


pip install matplotlib


# In[3]:


pip install wordcloud


# In[ ]:


#Import the necessary Python libraries and dataset 


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")

data = pd.read_csv("/Users/niquanlockhart/Downloads/tiktok_google_play_reviews.csv")
print(data.head())


# In[ ]:


#create a new dataset with just these two columns 


# In[8]:


data = data[["content", "score"]]
print(data.head())


# In[ ]:


#Created Two columns contain any null values


# In[9]:


print(data.isnull().sum())


# In[ ]:


#Cleaned text in the content column:


# In[10]:


stopword=set(stopwords.words('english'))
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["content"] = data["content"].apply(clean)


# In[ ]:


#Percentages of ratings given to TikTok on the Google Play store:


# In[11]:


ratings = data["score"].value_counts()
numbers = ratings.index
quantity = ratings.values
import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=numbers,hole = 0.5)
figure.show()


# In[ ]:


#Kinds of words the users use in the reviews of TikTok


# In[12]:


text = " ".join(i for i in data.content)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


#Added three more columns in this dataset as Positive, Negative, and Neutral by calculating the sentiment scores of the tweets


# In[13]:


nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["content"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["content"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["content"]]
data = data[["content", "Positive", "Negative", "Neutral"]]
print(data.head())


# In[ ]:


#Kinds of words people use in the positive reviews of TikTok


# In[14]:


positive =' '.join([i for i in data['content'][data['Positive'] > data["Negative"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(positive)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


#Kinds of words people often use in the negative reviews of TikTok


# In[15]:


negative =' '.join([i for i in data['content'][data['Negative'] > data["Positive"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(negative)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




