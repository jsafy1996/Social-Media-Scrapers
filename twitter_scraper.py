"""
This module contains classes for scraping, processing, and
saving tweets. Supports Twitter's Streaming API and REST API.
It is meant for NLP classification task data gathering, and
supports grouping by label.
"""

import json
import re
from time import sleep

import textblob
import tweepy
import numpy as np
import pandas as pd

from twitter_credentials import *


class TwitterClient:
    """
    Connect to the Twitter API and gather tweets.
    """

    def __init__(self, num_tweets, twitter_users=[None]):
        """
        Initialize member of class.

        Parameters
        ----------
        num_tweets (int): Number of tweets to scrape.
        twitter_users (list[list[str]]):
            List of lists where each inner list represents a
            label i.e. liberal/conservative, happy/sad, etc.
            Each inner list is populated with strings of user
            handles who belong in their respective group.
        """

        self.num_tweets = num_tweets
        self.twitter_users = twitter_users
        self.auth = self.authenticate_twitter_app()
        self.api = tweepy.API(self.auth)

    def get_api(self):
        """Return API object."""

        return self.api

    def authenticate_twitter_app(self):
        """Use twitter_credentials.py to login to Twitter API."""

        auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
        return auth

    def error_handler(self, cursor):
        """Handle Twitter API rate limits."""

        while True:
            try:
                yield cursor.next()
            except tweepy.RateLimitError:
                print(tweepy.RateLimitError)
                sleep(15 * 60)
            except:
                break

    def get_user_timeline_tweets(self):
        """
        Gather user-specified number of tweets from user-specified accounts
        using REST API. Attempt to gather the same number of tweets from
        each account, but if some accounts don't have enough tweets gather
        more from the next account in order to meet the user-specified
        number of tweets.

        Returns
        -------
        tweets (list[str]): List of tweets.
        """

        tweets = []
        carry = 0
        for user in self.twitter_users:
            oldlen = len(tweets)
            goal = self.num_tweets // len(self.twitter_users) + carry
            for tweet in self.error_handler(
                tweepy.Cursor(self.api.user_timeline, id=user).items(goal)
            ):
                tweets.append(tweet._json)
            carry = goal - (len(tweets) - oldlen)
            print(
                f"{self.twitter_users.index(user) + 1} out of {len(self.twitter_users)}"
            )
            
        return tweets

    def stream_tweets(self, hash_tag_list):
        """
        Gather user-specified number of tweets using streaming API.
        Use user-specified hash-tags to sort tweets into labels.

        Parameters
        ----------
        hash_tag_list (list[list[str]]):
            List of lists where each inner list represents a
            label i.e. liberal/conservative, happy/sad, etc.
            Each inner list is populated with hashtags which
            represent their respective group.

        Returns
        -------
        tweets (list[str]): List of tweets.
        """

        listener = TwitterListener(self.num_tweets)
        stream = tweepy.Stream(self.api.auth, listener)
        stream.filter(track=hash_tag_list, languages=["en"])
        tweets = listener.get_tweets()
        return tweets


class TwitterListener(tweepy.StreamListener):
    """
    Extend tweepy's StreamListener to override behavior
    upon receiving a tweet which matches the filter.
    """

    def __init__(self, num_tweets):
        """
        Initialize member of class.

        Parameters
        ----------
        num_tweets (int): Number of tweets to scrape.
        """
        self.num_tweets = num_tweets
        self.tweets = []

    def on_error(self, status):
        """Overrides StreamListener to handle errors."""
        print(status)
        if status == 420:
            return False

    def on_data(self, data):
        """Overrides StreamListener to append found data."""
        while len(self.tweets) < self.num_tweets:
            try:
                tweet = json.loads(data)
                # Avoid retweets to minimize duplicates.
                if not tweet["retweeted"]:
                    print(f"{len(self.tweets)} out of {self.num_tweets}") if len(
                        self.tweets
                    ) % 100 == 0 else 0
                    self.tweets.append(tweet)
            except BaseException as e:
                pass
            return True
        return False

    def get_tweets(self):
        """Return scraped tweets"""
        return self.tweets


class TweetAnalyzer:
    """
    Contains functions to clean, process, and save tweets.
    """

    def clean_tweet(self, tweet):
        """Remove any unusual characters from tweet (str) and return tweet (str)."""
        return " ".join(
            re.sub(
                "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet
            ).split()
        )

    def analyze_sentiment(self, tweet):
        """Return label (int) representing tweet (str) sentiment."""
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_data_frame(self, tweets, label):
        """
        Processes tweets and returns dataframe.

        Parameters
        ----------
        tweets (list[str]): List of tweets to handle.
        label (int): Desired label for this group of tweets.

        Returns
        -------
        (pandas.DataFrame): DataFrame containing tweets and features.
        """

        from nltk.corpus import stopwords
        import nltk

        df = pd.DataFrame(data=tweets, columns=["text"])

        df["sentiment"] = np.array(
            [self.analyze_sentiment(tweet) for tweet in df["text"]]
        )

        df["label"] = label

        nltk.download("stopwords")
        stops = stopwords.words("english")
        # We save processed text separately in case we ever need the originals.
        df["text_2"] = df["text"].apply(
            lambda x: " ".join([word for word in x.split() if word not in (stops)])
        )

        return df


def get_stream(total, filename):
    """
    Gather tweets from Streaming API. Note that total means
    total tweets per label, not total tweets, so a total
    of 50,000 means 100,000 tweets will be collected.

    Parameters
    ----------
    total (int): Total number of tweets to collect per label.
    filename (str): Where to write output.
    """

    df = pd.DataFrame()
    tags = [
        [],
        [],
    ]

    for i, label in enumerate(tags):
        twitter_client = TwitterClient(total)
        tweets = twitter_client.stream_tweets(label)
        tweet_analyzer = TweetAnalyzer()
        df_temp = df = tweet_analyzer.tweets_to_data_frame(tweets, i)
        df = df.append(df_temp, ignore_index=True)

        print(f"Done with side {i + 1} of {len(tags)}")

    df.to_csv(filename)


def get_users(total, filename):
    """
    Gather tweets from Streaming API. Note that total means
    total tweets per label, not total tweets, so a total
    of 50,000 means 100,000 tweets will be collected.

    Parameters
    ----------
    total (int): Total number of tweets to collect per label.
    filename (str): Where to write output.
    """

    df = pd.DataFrame()
    users = [
        [],
        [],
    ]

    for i, label in enumerate(users):
        twitter_client = TwitterClient(total, twitter_users=label)
        tweets = twitter_client.get_user_timeline_tweets()
        tweet_analyzer = TweetAnalyzer()
        df_temp = tweet_analyzer.tweets_to_data_frame(tweets, i)
        df = df.append(df_temp, ignore_index=True)

        print(f"Done with side {i + 1} of {len(users)}")

    df.to_csv(filename)


if __name__ == "__main__":
    get_users(50000, "./tweets_users.csv")
    get_stream(50000, "./tweets_stream.csv")
