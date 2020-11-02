# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import requests 
from requests_oauthlib import OAuth1
import os # OS was only used to bring in tokens from environment variables
import pandas as pd
import peter_sinkis_hw_twitter_api as hw


# hw.check_twitter()

# user_sinkis = hwf.find_user('@prsinkis')

# print(user_sinkis)

check_auth = hw.check_twitter()

print(check_auth)


user_sinkis = hw.find_user('@prsinkis')

print(user_sinkis)

test = hw.friends_of_friends(['@Beyonce', '@MariahCarey'], to_df=True)
print(test)


test2 = hw.friends_of_friends(['@Beyonce', '@MariahCarey'], keys=['id','name'], to_df=True)
print(test2)
