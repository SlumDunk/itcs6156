# %% import packages
import os
import re
import numpy as np
import pandas as pd
import itcs6156.settings as settings

# %% load dataset
file_path = os.path.join(settings.DATA_URL, 'listings.csv.gz')
df = pd.read_csv(file_path, compression='gzip')
print(df.info(verbose=True))

# %% preprocessing
# 1. time of host join Airbnb, processed to number days since first hosting in days
df_temp = (pd.to_datetime('2019-09-22') - pd.to_datetime(df['host_since'])).dt.days
df_select_X = df_temp.copy()

# 2. host response time in categories, processed to 4 dummy variables
# null value exists, not included in any category
df_temp = pd.get_dummies(df[['host_response_time']], prefix='hrt')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 3. host response rate in percentage, processed to float [0, 1]
# null value exists, not included in any category
df_temp = df['host_response_rate'].astype('str').map(lambda x: float(x.strip('%')) / 100)
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 4. whether host is superhost, processed to dummy variable 1-True 0-False
df_temp = (df['host_is_superhost'] == 't').astype('int')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 5. host listing count
df_select_X = pd.concat([df_select_X, df['host_listings_count']], axis=1)

# 6. whether host has profile picture, processed to dummy variable 1-True 0-False
df_temp = (df['host_has_profile_pic'] == 't').astype('int')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 7. whether host identity is verified, processed to dummy variable 1-True 0-False
df_temp = (df['host_identity_verified'] == 't').astype('int')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 8. host neighbourhood, processed to 26 dummy variables
df_temp = pd.get_dummies(df[['neighbourhood_cleansed']], prefix='nc')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 9. whether host location is exact, , processed to dummy variable 1-True 0-False
df_temp = (df['is_location_exact'] == 't').astype('int')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 10. property_type, processed to 22 dummy variables
df_temp = pd.get_dummies(df[['property_type']], prefix='pt')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 11. room type, , processed to 4 dummy variables
df_temp = pd.get_dummies(df[['room_type']], prefix='rt')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 12. maximum number of people the property accommodates
df_select_X = pd.concat([df_select_X, df['accommodates']], axis=1)

# 13. number of bathrooms available
df_select_X = pd.concat([df_select_X, df['bathrooms']], axis=1)

# 14. number of bedrooms available
df_select_X = pd.concat([df_select_X, df['bedrooms']], axis=1)

# 15. number of beds available
df_select_X = pd.concat([df_select_X, df['beds']], axis=1)

# 16. bed type, processed to 5 dummy variables
df_temp = pd.get_dummies(df[['bed_type']], prefix='bt')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 17. price per night in dollars
df_temp = df['price'].map(lambda x: re.compile(r'[^\d.]+').sub('', str(x))).astype('float')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 18. price per week in dollars
df_temp = df['weekly_price'].copy()
df_temp.loc[df['weekly_price'].isnull()] = 99999
df_temp = df_temp.map(lambda x: re.compile(r'[^\d.]+').sub('', str(x))).astype('float')
df_temp.loc[df_temp == 99999] = np.nan
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 19. price per month in dollars
df_temp = df['monthly_price'].copy()
df_temp.loc[df['monthly_price'].isnull()] = 99999
df_temp = df_temp.map(lambda x: re.compile(r'[^\d.]+').sub('', str(x))).astype('float')
df_temp.loc[df_temp == 99999] = np.nan
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 20. security deposit in dollars
df_temp = df['security_deposit'].copy()
df_temp.loc[df['security_deposit'].isnull()] = 99999
df_temp = df_temp.map(lambda x: re.compile(r'[^\d.]+').sub('', str(x))).astype('float')
df_temp.loc[df_temp == 99999] = np.nan
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 21. cleaning fee in dollars
df_temp = df['cleaning_fee'].copy()
df_temp.loc[df['cleaning_fee'].isnull()] = 99999
df_temp = df_temp.map(lambda x: re.compile(r'[^\d.]+').sub('', str(x))).astype('float')
df_temp.loc[df_temp == 99999] = np.nan
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 22. number of guests allowed in initial listing
df_select_X = pd.concat([df_select_X, df['guests_included']], axis=1)

# 23. additional fee for each additional guest in dollars
df_temp = df['extra_people'].map(lambda x: re.compile(r'[^\d.]+').sub('', str(x))).astype('float')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# 24. minimum number of nights needed for booking
df_select_X = pd.concat([df_select_X, df['minimum_nights']], axis=1)

# 25. minimum number of nights needed for booking
df_select_X = pd.concat([df_select_X, df['maximum_nights']], axis=1)

# 26. whether the host shows the availability of the listing, processed to dummy variable 1-True 0-False
df_temp = (df['has_availability'] == 't').astype('int')
df_select_X = pd.concat([df_select_X, df_temp], axis=1)

# %% target variables
# 1. number of days available for booking in the next 30, 60, 90, 365 days
df_select_Y = df[['availability_30', 'availability_60', 'availability_90', 'availability_365']].copy()

df_select_X.info()
