# Purpose: Twitter API Homework exercise
# Creator: Peter Sinkis
#
# Import statements used by the functions:
# import requests 
# from requests_oauthlib import OAuth1
# import os # OS was only used to bring in tokens from environment variables
# import pandas as pd
#
#

# All functions are set up to receive a python OAuth1 object as an argument
# These are defaulted to an argument of called: auth
# This manages twitter verification.
# 
# To test credentials use:
#
# API_PUBLIC = os.environ.get("TW_API_KEY")
# API_SECRET = os.environ.get("TW_API_SECRET_KEY")
# APP_PUBLIC = os.environ.get("TW_PRS_GA_HW_ACCESS_TOKEN")
# APP_SECRET = os.environ.get("TW_PRS_GA_HW_ACCESS_TOKEN_SECRET")
#
# url = 'https://api.twitter.com/1.1/account/verify_credentials.json'
# auth = OAuth1(
#             API_PUBLIC, #'USER_OAUTH_TOKEN', 
#             API_SECRET, #'USER_OAUTH_TOKEN_SECRET'
#             APP_PUBLIC, #'YOUR_APP_KEY', 
#             APP_SECRET #'YOUR_APP_SECRET',
#             )
#check_auth = requests.get(url, auth=auth)
#check_auth.reason should be 'ok'
#check_auth.status_code should be '200'

import requests 
from requests_oauthlib import OAuth1
import os # OS was only used to bring in tokens from environment variables
import pandas as pd



# Note for this file to work need to insert your keys here:
API_PUBLIC = 'x'
API_SECRET = 'x'
APP_PUBLIC = 'x'
APP_SECRET = 'x'

auth = OAuth1(API_PUBLIC, API_SECRET,APP_PUBLIC,APP_SECRET)


def check_twitter():
    url = 'https://api.twitter.com/1.1/account/verify_credentials.json'
    check_auth = requests.get(url, auth=auth)
    if check_auth.status_code == 200: 
        print('Credentials worked.')
    return check_auth


# Function 1

def find_user(screen_name,keys = [], auth=auth):
    # setup output dictionary
    output_dict = {}
    
    # Checks on screen_name input
    if screen_name[0] == '@':
        screen_name = screen_name[1:]

    # Go and get a result
    find_user_url = "https://api.twitter.com/1.1/users/search.json?q=" + screen_name
    req_output = requests.get(find_user_url, auth=auth)
    
    #convert the outut to a list, create an initial dictionary
    req_output_list = req_output.json()
    user_dict = {}
    i = 0
    for o in req_output_list:
        i = i+1
        user_dict.update({'user_' + str(i) : o})
    
    if not keys:
        output_dict = user_dict # don't filter the resulting dictionary
    else:
        for u in user_dict.keys():
            filtered_user_info_dict = {}
            for k in keys:
                filtered_user_info_dict.update({k : user_dict[u][k]})
            output_dict.update({u:filtered_user_info_dict})
                
    return output_dict



# Function 2

def find_hashtag(
    hashtag,
    count=10,
    search_type='popular', # Should also accept 'mixed' or 'recent'
    auth=auth):

    # Clean up hashtag input
    clean_hashtag = hashtag.replace('#','%23')
    clean_hashtag = clean_hashtag.replace(' ','%20')
    
    if clean_hashtag[0:3] != '%23':
        clean_hashtag = '%23'+clean_hashtag
    
    if search_type not in ['popular','mixed','recent']:
        raise Exception("Error: if specify search_type specify 'popular','mixed','recent'")
    
    # Set up search URL
    search_url = (
        "https://api.twitter.com/1.1/search/tweets.json?" 
        + "q=" + clean_hashtag
        + "&result_type=" + search_type
        + "&count=" + str(count)
        )
    print(search_url)
    query_output = requests.get(search_url, auth=auth)
    output_list = query_output.json()['statuses']
    
    return output_list



# Function 3

def get_followers(
        screen_name,
        keys = [
                'name',
                'followers_count',
                'friends_count',
                'screen_name'
                ], 
        to_df = False,
        auth=auth
        ):

    # Checks on screen_name input
    if screen_name[0] == '@':
        screen_name = screen_name[1:]

    # Go and get results, adding users to an overall list
    user_list = []
    cursor = -1
    while cursor != 0:
        find_follower_url = ("https://api.twitter.com/1.1/followers/list.json" 
                         + "?screen_name=" + screen_name
                         + "&cursor=" + str(cursor)
                         + "&count=200" 
                        )
        #print(f'find_user_url: {find_user_url}')
        follower_output = requests.get(find_follower_url, auth=auth)
        if follower_output.status_code == 429:
            print("Reached max API requests.\n"
                  + "Setting cursor to 0 to exit loop.\n"
                  + "Followers found so far will be output."
                 )
            cursor = 0
        else:
            follower_output_dict = follower_output.json()
            cursor = follower_output_dict['next_cursor']
            [user_list.append(user) for user in follower_output_dict['users']]
        
    # Limit the user_list to the specific keys required
    user_list_keys = []
    for user in user_list:
        temp_dict = {}
        for k in keys:
            temp_dict.update({k:user[k]})
        user_list_keys.append(temp_dict)
    
    # Allow for pandas data frame
    if to_df == False:
        return_object = user_list_keys
    elif to_df == True:
        # Loop through each potential column, and add to dictionary
        prep_df_dict = {}
        
        for k in keys:
            temp_list = [i[k] for i in user_list_keys]
            prep_df_dict.update({k:temp_list})
        
        #Add data to the data frame
        user_df = pd.DataFrame(data=prep_df_dict)
        
        return_object = user_df
    else:
        return_object = 'to_df argument only accpets True or False'
                
    return return_object



# Function 4 
# (note that I incorporated cursor management before realising that was part of Function 5)

def friends_of_friends(
    names,
    keys = [], 
    to_df = False,
    auth=auth
    ):
    
    if len(names) != 2:
        print("You can only enter two names as a list.")
    
    # Checks on screen_name input
    if names[0][0] == '@':
        names[0] = names[0][1:]    

    if names[1][0] == '@':
        names[1] = names[1][1:]       

        
    # Get overall list for first name
    # Go and get results, adding users to an overall list
    friends_list_0 = []
    cursor = -1
    while cursor != 0:
        find_friends_0_url = (
                        'https://api.twitter.com/1.1/friends/list.json'
                         + "?screen_name=" + names[0]
                         + "&cursor=" + str(cursor)
                         + "&count=200" 
                        )
        friends_0_output = requests.get(find_friends_0_url, auth=auth)
        if friends_0_output.status_code == 429:
            print("Reached max API requests.\n"
                  + "Setting cursor to 0 to exit loop.\n"
                  + "friends_0s found so far will be output."
                 )
            cursor = 0
        else:
            friends_0_output_dict = friends_0_output.json()
            cursor = friends_0_output_dict['next_cursor']
            [friends_list_0.append(user) for user in friends_0_output_dict['users']]

            
    # Get overall list for second name
    friends_list_1 = []
    cursor = -1
    while cursor != 0:
        find_friends_1_url = (
                        'https://api.twitter.com/1.1/friends/list.json'
                         + "?screen_name=" + names[1]
                         + "&cursor=" + str(cursor)
                         + "&count=200" 
                        )
        friends_1_output = requests.get(find_friends_1_url, auth=auth)
        if friends_1_output.status_code == 429:
            print("Reached max API requests.\n"
                  + "Setting cursor to 0 to exit loop.\n"
                  + "friends_1s found so far will be output."
                 )
            cursor = 0
        else:
            friends_1_output_dict = friends_1_output.json()
            cursor = friends_1_output_dict['next_cursor']
            [friends_list_1.append(user) for user in friends_1_output_dict['users']]
    
    # Get list of ids for both sets of friends
    name_0_friends_ids = [i['id'] for i in friends_list_0]
    name_1_friends_ids = [i['id'] for i in friends_list_1]

    # Find common ids
    common_friends_list = []
    if len(name_0_friends_ids) < len(name_1_friends_ids):
        common_friend_ids = [i for i in name_0_friends_ids if i in name_1_friends_ids]
        common_friends_list = [i for i in friends_list_0 if i['id'] in common_friend_ids]
        
    else:
        common_friend_ids = [i for i in name_1_friends_ids if i in name_0_friends_ids]
        common_friends_list = [i for i in friends_list_1 if i['id'] in common_friend_ids]
        
    # Limit the friends_list_0 to the specific keys required
    common_friends_list_keys = []
    
    if not keys:
        common_friends_list_keys = common_friends_list
    else:
        for user in common_friends_list:
            temp_dict = {}
            for k in keys:
                temp_dict.update({k:user[k]})
            common_friends_list_keys.append(temp_dict)    
    
    # Put results into a pandas dataframe
    
    
    # Allow for pandas data frame
    if to_df == False:
        return_object = common_friends_list_keys
    elif to_df == True:
        # Loop through each potential column, and add to dictionary
        prep_df_dict = {}
        
        # Check if keys have been designated, if not add them back
        if not keys:
            keys_found = []
            for u in common_friends_list_keys:
                for k in u.keys():
                    keys_found.append(k)
            keys = list(set(keys_found)) # dedupe the keys
            keys.sort()
            
        # Set up the columns
        for k in keys:
            temp_list = [i.get(k,'No value returned.') for i in common_friends_list_keys]
            prep_df_dict.update({k:temp_list})
        
        #Add data to the data frame
        user_df = pd.DataFrame(data=prep_df_dict)
        
        return_object = user_df
    else:
        return_object = 'to_df argument only accpets True or False'
                
    return return_object



# Function 5
# Added variable required for Function 5,
# called it friend_of_friends_fs to distinguish it from function 4

def friends_of_friends_fs(
    names,
    keys = [], 
    to_df = False,
    full_search=True,
    auth=auth
    ):
    
    if len(names) != 2:
        print("You can only enter two names as a list.")
    
    # Checks on screen_name input
    if names[0][0] == '@':
        names[0] = names[0][1:]    

    if names[1][0] == '@':
        names[1] = names[1][1:]       

        
    # Get overall list for first name
    # Go and get results, adding users to an overall list
    friends_list_0 = []
    cursor = -1
    while cursor != 0:
        find_friends_0_url = (
                        'https://api.twitter.com/1.1/friends/list.json'
                         + "?screen_name=" + names[0]
                         + "&cursor=" + str(cursor)
                         + "&count=200" 
                        )
        friends_0_output = requests.get(find_friends_0_url, auth=auth)
        if friends_0_output.status_code == 429:
            print("Reached max API requests.\n"
                  + "Setting cursor to 0 to exit loop.\n"
                  + "friends_0s found so far will be output."
                 )
            cursor = 0
        ##############
        # Modification for function 5
        elif full_search == False:
            cursor = 0 # If have full search turned off set cursor to 0 to exit loop after 1 pass
        ###############
        else:
            friends_0_output_dict = friends_0_output.json()
            cursor = friends_0_output_dict['next_cursor']
            [friends_list_0.append(user) for user in friends_0_output_dict['users']]

            
    # Get overall list for second name
    friends_list_1 = []
    cursor = -1
    while cursor != 0:
        find_friends_1_url = (
                        'https://api.twitter.com/1.1/friends/list.json'
                         + "?screen_name=" + names[1]
                         + "&cursor=" + str(cursor)
                         + "&count=200" 
                        )
        friends_1_output = requests.get(find_friends_1_url, auth=auth)
        if friends_1_output.status_code == 429:
            print("Reached max API requests.\n"
                  + "Setting cursor to 0 to exit loop.\n"
                  + "friends_1s found so far will be output."
                 )
            cursor = 0
        ##############
        # Modification for function 5
        elif full_search == False:
            cursor = 0 # If have full search turned off set cursor to 0 to exit loop after 1 pass
        ###############
        else:
            friends_1_output_dict = friends_1_output.json()
            cursor = friends_1_output_dict['next_cursor']
            [friends_list_1.append(user) for user in friends_1_output_dict['users']]
    
    # Get list of ids for both sets of friends
    name_0_friends_ids = [i['id'] for i in friends_list_0]
    name_1_friends_ids = [i['id'] for i in friends_list_1]

    # Find common ids
    common_friends_list = []
    if len(name_0_friends_ids) < len(name_1_friends_ids):
        common_friend_ids = [i for i in name_0_friends_ids if i in name_1_friends_ids]
        common_friends_list = [i for i in friends_list_0 if i['id'] in common_friend_ids]
        
    else:
        common_friend_ids = [i for i in name_1_friends_ids if i in name_0_friends_ids]
        common_friends_list = [i for i in friends_list_1 if i['id'] in common_friend_ids]
        
    # Limit the friends_list_0 to the specific keys required
    common_friends_list_keys = []
    
    if not keys:
        common_friends_list_keys = common_friends_list
    else:
        for user in common_friends_list:
            temp_dict = {}
            for k in keys:
                temp_dict.update({k:user[k]})
            common_friends_list_keys.append(temp_dict)    
    
    # Put results into a pandas dataframe
    
    
    # Allow for pandas data frame
    if to_df == False:
        return_object = common_friends_list_keys
    elif to_df == True:
        # Loop through each potential column, and add to dictionary
        prep_df_dict = {}
        
        # Check if keys have been designated, if not add them back
        if not keys:
            keys_found = []
            for u in common_friends_list_keys:
                for k in u.keys():
                    keys_found.append(k)
            keys = list(set(keys_found)) # dedupe the keys
            keys.sort()
            
        # Set up the columns
        for k in keys:
            temp_list = [i.get(k,'No value returned.') for i in common_friends_list_keys]
            prep_df_dict.update({k:temp_list})
        
        #Add data to the data frame
        user_df = pd.DataFrame(data=prep_df_dict)
        
        return_object = user_df
    else:
        return_object = 'to_df argument only accpets True or False'
                
    return return_object
