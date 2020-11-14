+++
title = "Reddit Comment Analysis"
author = ["Sunny Hasija"]
date = 2020-11-14T14:28:00-05:00
draft = false
tags = ""
subtitle = "What can Reddit Comments Tell us"
featured = true
categories = ""
highlight = true
+++

We have already scraped and stored the reddit comments from /r/CouriersofReddit, /r/PostMates, and /r/DoorDash. The data is now in a pickle file, and is of the following format:

-   PK: Primary Key - the reddit post ID
-   Title: Title of the post.
-   URL: link to the post.
-   Time: time of creation of the post.
-   Comments: the particular comment.
-   Comment\_Time: The time the comment was created.
-   Is\_Root\_Comment: Identifies if the comment is a root comment in the post submission.
-   Comment\_Parent: What which comment is the parent of this comment, has the comment ID in it.
-   Comment\_PK: the comment ID.

Import all the libraries that are needed - and read in the pickle into a data frame.

```python
import pandas as pd
import numpy as np
import matplotlib
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

df = pd.read_pickle("/home/cantos/Dropbox/School/My Papers in Progress/Crowdsourced Delivery/flat-comment-df.pkl")
df=df.drop_duplicates()
```

We might need to save the file as a TSV for sharing with others. The TSV preserves the commas in the comments, and can also be ingested by Excel, R, and Tableau.

```python

#writefile = df.to_csv('/home/cantos/Dropbox/School/My Papers in Progress/Crowdsourced Delivery/flat-comment.tsv', sep='\t')
```


## Searching the corpus for keywords {#searching-the-corpus-for-keywords}

Ideally, we want to be able to create a smaller dataset for analysis from this big data set. We create an array with all the search terms we want to search for, and then assign the search result into another data frame. We can then print out how many posts and comments are present in the resulting dataset.

```python

search_terms = ['race', 'racism', 'racist', 'discriminate']
searchResult = df[df['Comments'].str.contains('|'.join(search_terms))] #creates the result dataframe.
print("The number of comments with search terms:", len(searchResult.index))
print("Total number of Posts", len(pd.unique(searchResult['PK'])))
```

```text
The number of comments with search terms: 1147
Total number of Posts 583
```


### Top 10 posts in the dataset {#top-10-posts-in-the-dataset}

After we have created the dataset, it is evident from the last step that there are 1147 comments in 583 posts. This indicates that there might be some posts with very few comments, and some with many. Let's get a list of the top 10 posts with the most comments.

```python

topPosts = searchResult['Title'].value_counts()[:10].index.tolist()
for post in topPosts:
    print(post)
```

```text
I feel like this promotes racism rather than helps..
Deactivated Because a Racist manager not only called the police on me, but also reported me to Doordash
Postmates driver encounters deranged woman
The MPLS cop that killed George Floyd can't get doordash delivered.
I hit my 1000th delivery today. A few thoughts about my experiences.
Never thought I'd get one of these...
When the $3 orders can no longer keep you afloat
Courier (rightfully) fears for his life
This is why your food is cold
So I was delivering the other day...
```


### Save the dataset {#save-the-dataset}

We might want to save this dataset as a pickle and a TSV if needed. Lets do that.

```python

writefile = searchResult.to_csv('/home/cantos/Dropbox/School/My Papers in Progress/Crowdsourced Delivery/search-result-flat-comment.tsv', sep='\t')
## Write the file to pickle for other scripts to use.
searchResult.to_pickle("/home/cantos/Dropbox/School/My Papers in Progress/Crowdsourced Delivery/search-result-flat-comment.pkl")
```


## Word Cloud: {#word-cloud}

I want to evaluate quickly what kind of themes are in the present in the data set. A word cloud usually works pretty well. I think because we are using the terms "race/ism/ist" in the search,  I would probably want to remove those words along with other stop words.

First thing we do is to put all the text in the comments into one variable, and look at how many words there are in total.

```python

text="".join(comment for comment in searchResult.Comments)
print ("There are {} words in the combination of all comments.".format(len(text)))
stopwords = set(STOPWORDS)
stopwords.update(["racist", "racism", "race", "people", "will","still"])
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
```

Show the wordcloud, and save it to a file

```python

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("/home/cantos/Dropbox/School/My Papers in Progress/Crowdsourced Delivery/wordcloud-search-comments.png")
```


## Sentiment Analysis {#sentiment-analysis}

Although there are multiple types of sentiment analysis engines out there, I do not want to use a custom training paradigm, or me having to train the model myself (trying to save some time and computer cycles). Moreover, there have been recent developments with sentiment analysis that I want to leverage. Enter VADER.


### VADER {#vader}

VADER stands for Valence Aware Dictionary sEntiment Reasoner, and is a lexicon and rule-based sentiment analysis tool that is **specifically attuned to sentiments expressed in social media**. This makes it particularly useful for analysing reddit comments. VADER uses a combination of a sentiment lexicon - which is a list of lexical features (words) which are generally labelled according to their semantic orientation as either positive or negative, and also the polarity of the positive and negative sentiment.

VADER was developed using Amazon's Mechanical Turk platform to get most of their ratings, and can be considered the gold standard of sentiment lexicons.


### Advantages of VADER {#advantages-of-vader}

-   works exceedingly well for social media text, yet readily generalizes to other domains.
-   doesn't require any training data - its created using generalizable, valence based, human curated sentiment lexicon
-   fast : you can even use this with streaming data
-   It does not suffer from a speed-performance tradeoff


### Implementing VADER {#implementing-vader}

You can download vader via pip, using the following command
`pip3 install vaderSentiment`

We then want to create a new dataframe, I could do it in existing dataframe, but I am trying to not get it corrupted. Create the new DF with these additional columns `Comment_Neg,Comment_Neu, Comment_Pos, Comment_Comp`.

We can then take the sentiment of each comment in the dataframe, and add all the information back to the new data frame.

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

column_names = ["PK","Title", "URL", "Time", "Comment_PK", "Comments", "Comment_Time", "Comment_Score", "Is_Root_Comment","Comment_Parent", "Comment_Neg","Comment_Neu", "Comment_Pos", "Comment_Comp"]

postSentimentDF=pd.DataFrame(columns=column_names)

sia_obj = SentimentIntensityAnalyzer()  #create a SentimentIntensityAnalyzer object
for index,comment in searchResult.iterrows():
    sentiment_dictionary = sia_obj.polarity_scores(comment["Comments"])
    postSentimentDF = postSentimentDF.append({'PK':comment['PK'], 'Title':comment['Title'], 'URL':comment['URL'], 'Time':comment['Time'], "Comments":comment['Comments'], "Comment_Time":comment['Comment_Time'], "Comment_Score":comment['Comment_Score'], "Is_Root_Comment":comment['Is_Root_Comment'], "Comment_Parent":comment['Comment_Parent'], 'Comment_PK':comment['Comment_PK'], 'Comment_Neg':sentiment_dictionary['neg'], 'Comment_Neu':sentiment_dictionary['neu'], 'Comment_Pos':sentiment_dictionary['pos'], 'Comment_Comp':sentiment_dictionary['compound']},ignore_index=True )
    #print(comment)

print(postSentimentDF.head())
```

```text
       PK                                              Title                     URL                 Time Comment_PK  ... Comment_Parent Comment_Neg Comment_Neu Comment_Pos Comment_Comp
0  fshgt1                                                Lol  https://redd.it/fshgt1  2020-03-31 17:45:09    fm3gjaj  ...      t3_fshgt1       0.298       0.702       0.000      -0.9163
1  gbzdoc  Well everyone. My first controversial delivery...  https://redd.it/gbzdoc  2020-05-02 03:14:19    fp8kqba  ...     t1_fp8dz0u       0.100       0.814       0.086      -0.0772
2  gbzdoc  Well everyone. My first controversial delivery...  https://redd.it/gbzdoc  2020-05-02 03:14:19    fp968s1  ...     t1_fp8p0fs       0.000       0.860       0.140       0.8923
3  hjgrtz                        I was hacked, $338 was lost  https://redd.it/hjgrtz  2020-07-01 19:07:32    fwme4y4  ...      t3_hjgrtz       0.025       0.930       0.045       0.2732
4  hjgrtz                        I was hacked, $338 was lost  https://redd.it/hjgrtz  2020-07-01 19:07:32    fwmg9a5  ...     t1_fwme4y4       0.038       0.962       0.000      -0.2960

[5 rows x 14 columns]
```

Let us go ahead and save this file in pickle and TSV format.

```python
writefile = postSentimentDF.to_csv('/home/cantos/Dropbox/School/My Papers in Progress/Crowdsourced Delivery/sentiment-search-result-flat-comment.tsv', sep='\t')
## Write the file to pickle for other scripts to use.
postSentimentDF.to_pickle("/home/cantos/Dropbox/School/My Papers in Progress/Crowdsourced Delivery/sentiment-search-result-flat-comment.pkl")
```
