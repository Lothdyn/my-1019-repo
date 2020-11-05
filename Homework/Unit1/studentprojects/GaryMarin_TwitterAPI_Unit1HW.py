import pandas as pd
import requests
from requests_oauthlib import OAuth1

YOUR_APP_KEY = "UxGtf03NrSZhXVSCM39JKBwn3"
YOUR_APP_SECRET = "Y0H595UaKWTOtnnpbhCCvqJiq3JuFJ3eB68TEut1L7o7JtCz0W"
USER_OAUTH_TOKEN = "307856162-UGrzbFTyTDzl3SW1Juhj1V45ZiATOi94V7VS0nej"
USER_OAUTH_TOKEN_SECRET = "B3bmdco44Xy14dHjh6gF2xD84MKTNHUZF9UDt4WQrHK8P"

def find_user(screen_name, keys=[]):
    user_dict = {}

    #cleans up username string
    username = screen_name.replace('@','')
    url = f'https://api.twitter.com/1.1/users/lookup.json?screen_name={username}'
    auth = OAuth1(YOUR_APP_KEY, YOUR_APP_SECRET, USER_OAUTH_TOKEN, USER_OAUTH_TOKEN_SECRET)
    req = requests.get(url, auth=auth).json()
    
    if len(keys)==0:
        user_dict = req[0]
        return user_dict
    else:
        for key in keys:
            user_dict[key] = req[0][key]
        return user_dict


#is there a customary default value that should be used for count?
#using popular as default search_type

def find_hashtag(searchterm, count = 3 , search_type = 'popular'):
    
    #adjust searchterm for url
    if searchterm[0] == "#":
        searchterm = searchterm.replace("#","%23")
    else:
        searchterm = "%23" + searchterm
    
    #adjust search_type for url
    if 'recent' in search_type:
        search_type = 'recent'
    elif 'mixed' in search_type:
        search_type = 'mixed'
    else:
        search_type = 'popular'
    
    
    url = f'https://api.twitter.com/1.1/search/tweets.json?q={searchterm}&count={str(count)}&search_type={search_type}'
    #print(url) #print to make sure url looks how we want it to
    auth = OAuth1(YOUR_APP_KEY, YOUR_APP_SECRET, USER_OAUTH_TOKEN, USER_OAUTH_TOKEN_SECRET)
    req = requests.get(url, auth=auth).json()
    return req['statuses']

#does not work entirely, sometimes the return is a list instead of a dict of objects
def get_followers(screen_name, keys=[],to_df=False):
    #dict to return or be used for data frame
    user_dict = {}
    
    username = screen_name.replace('@','')
    url = f'https://api.twitter.com/1.1/followers/list.json?screen_name={screen_name}'
    auth = OAuth1(YOUR_APP_KEY, YOUR_APP_SECRET, USER_OAUTH_TOKEN, USER_OAUTH_TOKEN_SECRET)
    req = requests.get(url, auth=auth).json()
    
    for key in keys:
        user_dict[key] = [user[key] for user in req['users']] #list comprehension to get return dict ready
        #print(user_dict)
        #print('couldnt get this data') #was attached to an if/else

    #creates a data frame if parameter was true else return the dict of objects
    if to_df == True:
        df = pd.DataFrame(user_dict)
        return df
    else:
        return user_dict





    
