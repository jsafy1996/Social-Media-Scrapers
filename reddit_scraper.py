"""
This module contains classes for scraping, processing, and
saving reddit posts and comments. It is meant for NLP
classification task data gathering, and supports grouping by label.
"""

import json

import praw
import textblob
import numpy as np
import pandas as pd

from reddit_credentials import *


class Scraper:
    """
    Connect to the Reddit API and gather posts and comments.
    """

    def __init__(self, num_posts, subreddits):
        """
        Initialize member of class.

        Parameters
        ----------
        num_posts (int): Number of posts to scrape.
        subreddits (list[str]): List of subreddits to search.
        """

        self.subreddits = subreddits
        self.num_posts = num_posts
        self.reddit = self.connect(API_KEY, API_SECRET, USER_AGENT)

    def connect(self, API_KEY, API_SECRET, USER_AGENT):
        """Return API object."""

        reddit = praw.Reddit(
            client_id=API_KEY, client_secret=API_SECRET, user_agent=USER_AGENT
        )
        return reddit

    def scrape_posts(self):
        """
        Gather user-specified number of posts from user-specified subreddits
        Attempt to gather the same number of posts from each account, but if
        some accounts don't have enough posts gather more from the next
        account in order to meet the user-specified number of posts.

        Returns
        -------
        posts (list[str]): List of posts.
        """

        posts = []
        carry = 0
        for subreddit in self.subreddits:
            oldlen = len(posts)
            goal = self.num_posts // len(self.subreddits) + carry
            for post in self.reddit.subreddit(subreddit).top(limit=goal):
                posts.append(post.title)
            carry = goal - (len(posts) - oldlen)
            print(
                f"{self.subreddits.index(subreddit) + 1} out of {len(self.subreddits)}"
            )

        return posts

    def scrape_comments(self):
        """
        Gather user-specified number of comments from user-specified subreddits
        Attempt to gather the same number of comments from each account, but if
        some accounts don't have enough comments gather more from the next
        account in order to meet the user-specified number of comments.

        Returns
        -------
        comments (list[str]): List of comments.
        """

        comments = []
        carry_1 = 0
        carry_2 = 0
        # Divide the term in parentheses by the number of comments you want per post.
        posts_to_search = (self.num_posts // len(self.subreddits)) // 5
        for subreddit in self.subreddits:
            oldlen_1 = len(comments)
            goal_1 = self.num_posts // len(self.subreddits) + carry_1 # Goal per subreddit
            for post in self.reddit.subreddit(subreddit).top(
                time_filter="year", limit=posts_to_search
            ):
                oldlen_2 = len(comments)
                goal_2 = goal_1 // posts_to_search + carry_2 # Goal per post
                post.comments.replace_more(limit=0)
                for comment in post.comments:
                    if len(comments) == oldlen_2 + goal_2:
                        break
                    comments.append(comment.body)
                carry_2 = goal_2 - (len(comments) - oldlen_2)
            carry_1 = goal_1 - (len(comments) - oldlen_1)
            print(
                f"{self.subreddits.index(subreddit) + 1} out of {len(self.subreddits)}"
            )
        return comments


class RedditAnalyzer:
    """
    Contains functions to clean, process, and save reddit posts and comments.
    """

    def clean_post(self, post):
        """Remove any unusual characters from post (str) and return post (str)."""
        return " ".join(
            re.sub(
                "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", post
            ).split()
        )

    def analyze_sentiment(self, post):
        """Return label (int) representing post (str) sentiment."""
        analysis = TextBlob(self.clean_post(post))

        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def posts_to_data_frame(self, posts, label):
        """
        Processes posts and returns dataframe.

        Parameters
        ----------
        posts (list[str]): List of posts to handle.
        label (int): Desired label for this group of posts.

        Returns
        -------
        (pandas.DataFrame): DataFrame containing posts and features.
        """

        from nltk.corpus import stopwords
        import nltk

        df = pd.DataFrame(data=posts, columns=["text"])

        df["sentiment"] = np.array(
            [self.analyze_sentiment(post) for post in df["text"]]
        )

        df["label"] = label

        nltk.download("stopwords")
        stops = stopwords.words("english")
        # We save processed text separately in case we ever need the originals.
        df["text_2"] = df["text"].apply(
            lambda x: " ".join([word for word in x.split() if word not in (stops)])
        )

        return df


def get_posts(total, filename):
    """
    Gather posts from Streaming API. Note that total means
    total posts per label, not total posts, so a total
    of 50,000 means 100,000 posts will be collected.

    Parameters
    ----------
    total (int): Total number of posts to collect per label.
    filename (str): Where to write output.
    """

    subs = [
        [],
        [],
    ]

    df = pd.DataFrame()

    for i, label in enumerate(subs):
        scraper = Scraper(total, label)
        posts = scraper.scrape_posts()
        analyzer = RedditAnalyzer()
        df_temp = analyzer.posts_to_data_frame(posts, i)
        df = df.append(df_temp, ignore_index=True)
        print(f"Done with side {i + 1} of {len(subs)}")

    df.to_csv(filename)


def get_comments(total, filename):
    """
    Gather comments from Streaming API. Note that total means
    total comments per label, not total comments, so a total
    of 50,000 means 100,000 comments will be collected.

    Parameters
    ----------
    total (int): Total number of comments to collect per label.
    filename (str): Where to write output.
    """

    subs = [
        [],
        [],
    ]

    df = pd.DataFrame()

    for i, label in enumerate(subs):
        scraper = Scraper(total, label)
        comments = scraper.scrape_comments()
        analyzer = RedditAnalyzer()
        df_temp = analyzer.posts_to_data_frame(comments, i)
        df = df.append(df_temp, ignore_index=True)
        print(f"Done with side {i + 1} of {len(subs)}")

    df.to_csv(filename)


if __name__ == "__main__":
    get_posts(50000, "./reddit_posts.csv")
    get_comments(50000, "./reddit_comments.csv")