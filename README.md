# Social-Media-Scrapers

This repo contains modules to help speed up data collection. Each module helps collect, process, and save data pertaining to a social network. They are meant for NLP classification task data gathering, and support grouping by label.

So far this repo supports:
1. Twitter
2. Reddit

To use, enter your API credentials for the network you want to scrape in $NETWORK_credentials.py. Then edit the corresponding script's "get_x" functions with the labels you would like to scrape. For example, if you want to build a liberal / conservative classifier with Twitter data from known liberal / conservative users you can edit the users variable in get_users to users = [["Conservative", "Users", "Here"], ["Liberal", "Users", "Here"]].

Hope these tools help you with your NLP research!