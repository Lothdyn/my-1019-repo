import pandas as pd
import requests
from requests_oauthlib import OAuth1
##AlvaroGHW


##url_followers = f'https://api.twitter.com/1.1/followers/list.json?screen_name={username}'
##url_usersname = f'https://api.twitter.com/1.1/users/search.json?q={username}'
auth = OAuth1('xbeKBFLPJXaZYRUc8hNqcEyx6', 'mOZ9PddlArqVEJDg4FOjk8hO6pSgXxs3bHxjOA8oLTYBeXozN4',
               '710862333506146304-bQm967t61naDhBlXOwqAoVnS12VUzOb', 'Q1kB7EhrLKtIknxcu62vkc9fjPnhLnP2IuXYQkJf15fM6')



#user_info start
def find_user(username):
    if username[0] == '#':
        username = username[1:]
        username
    req = requests.get(f'https://api.twitter.com/1.1/users/search.json?q={username}',auth=auth)
    data = req.json()
    user_info = data[0]
    user_info_key = [dict((k, user_info[k]) for k in ('name', 'screen_name', 'followers_count','friends_count'))]
    print(user_info_key)
    
find_user('alvarogz01')

#%%


def  find_hashtag(hashtag):
        if hashtag[0] == '#':
            hashtag = hashtag[1:]
            hashtag
        req = requests.get(f'https://api.twitter.com/1.1/search/tweets.json?q=%23{hashtag}',auth=auth)
        data = req.json()
        hashtag_tweets = pd.DataFrame(data['statuses'])
        print(hashtag_tweets)
                
        
find_hashtag('#DataScience')        
        
#%%%  I know you wanted this in one functions but I will need to see the solutions to see how that is done. google is unhelpful

def  find_hashtag(hashtag):
        if hashtag[0] == '#':
            hashtag = hashtag[1:]
            hashtag
        req = requests.get(f'https://api.twitter.com/1.1/search/tweets.json?q=%23{hashtag}',auth=auth)
        data = req.json()
        hashtag_tweets = pd.DataFrame(data['statuses'])
        print(hashtag_tweets)
                
        
find_hashtag('#DataScience')  

#hashtag = 'DataScience'
#url_hashtag   = f'https://api.twitter.com/1.1/search/tweets.json?q=%23{hashtag}'
#url_hashtagcount  = f'https://api.twitter.com/1.1/search/tweets.json?q=%23{hashtag}&count={count}'
##url_hashtagtype   = f'https://api.twitter.com/1.1/search/tweets.json?q=%23{hashtag}&result_type={type}'
#req = requests.get(url_hashtagcount, auth=auth)
#data = req.json()
#data['statuses']
#hashtag_tweets = pd.DataFrame(data['statuses'])[['created_at', 'text']]
#%%
def  find_hashtag1(hashtag, count):
        if hashtag[0] == '#':
            hashtag = hashtag[1:]
            hashtag
        req = requests.get(f'https://api.twitter.com/1.1/search/tweets.json?q=%23{hashtag}&count={count}',auth=auth)
        data = req.json()
        hashtag_tweets = pd.DataFrame(data['statuses'])
        print(hashtag_tweets)
                
        
find_hashtag1('#DataScience', 50) 

#%%
def  find_hashtag2(hashtag, type):
        if hashtag[0] == '#':
            hashtag = hashtag[1:]
            hashtag
        req = requests.get(f'https://api.twitter.com/1.1/search/tweets.json?q=%23{hashtag}&result_type={type}',auth=auth)
        data = req.json()
        hashtag_tweets = pd.DataFrame(data['statuses'])
        print(hashtag_tweets)
                
        
find_hashtag2('#DataScience', 'mixed') 

#%%
def find_followers(username):
     if username[0] == '#':
        username = username[1:]
        username
     req = requests.get( f'https://api.twitter.com/1.1/followers/list.json?screen_name={username}', auth=auth)
     data = req.json()
     names = [item['name'] for item in data['users']]
     followers_count = [item['followers_count'] for item in data['users']]
     friends_count = [item['friends_count'] for item in data['users']]
     screenname = [item['screen_name'] for item in data['users']]
     keys = {
    'name': names,
    'FollowersCount': followers_count,
    'FriendsCount': friends_count,
    'ScreenName': screenname}
     print(pd.DataFrame(keys))
         
    
find_followers('alvarogz01')

#%%

def find_followers1(username, to_df):
     if username[0] == '#':
        username = username[1:]
        username
     req = requests.get( f'https://api.twitter.com/1.1/followers/list.json?screen_name={username}', auth=auth)
     data = req.json()
     names = [item['name'] for item in data['users']]
     followers_count = [item['followers_count'] for item in data['users']]
     friends_count = [item['friends_count'] for item in data['users']]
     screenname = [item['screen_name'] for item in data['users']]
     keys = {
    'name': names,
    'FollowersCount': followers_count,
    'FriendsCount': friends_count,
    'ScreenName': screenname}
     if to_df == "True":
         print(pd.DataFrame(keys))
     print(data['users'])

find_followers1('alvarogz01', 'false')         
## Pretty sure each question was supposed to have just one function and I think I mis-understood Keys part to 1 and 3. 
## I want to submitt something but will continue to work on this regardless