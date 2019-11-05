# %% import packages
import pandas as pd

# %% load dataset
df = pd.read_csv('/Users/liuzhongda/PycharmProjects/itcs6156/static/dataset/listings_source.csv')
# print(df.info())
df.info(verbose=True)

# %% preprocessing
# 1. time of host join Airbnb, processed to number days since first hosting in days
df_temp = (pd.to_datetime('2019-09-22') - pd.to_datetime(df['host_since'])).dt.days
df_select_0 = df_temp

# 2. host response time in categories, processed to 4 dummy variables
# null value exists, not included in any category
df_temp = pd.get_dummies(df[['host_response_time']], prefix='hrt_')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 3. host response rate in percentage, processed to float [0, 1]
# null value exists, not included in any category
df_temp = df['host_response_rate'].astype('str').map(lambda x: float(x.strip('%')) / 100)
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 4. whether host is superhost, processed to dummay variable 1-True 0-False
df_temp = (df['host_is_superhost'] == 't').astype('int')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 5. host listing count
df_select_0 = pd.concat([df_select_0, df['host_listings_count']], axis=1)

# 6. whether host has profile picture, processed to dummay variable 1-True 0-False
df_temp = (df['host_has_profile_pic'] == 't').astype('int')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 7. whether host identity is verfied, processed to dummay variable 1-True 0-False
df_temp = (df['host_identity_verified'] == 't').astype('int')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 8. host neighbourhood in categories, processed to 26 dummy variables
df_temp = pd.get_dummies(df[['neighbourhood_cleansed']], prefix='nc_')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 9. whether host location is exact, , processed to dummay variable 1-True 0-False
df_temp = (df['is_location_exact'] == 't').astype('int')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 10. property_type, processed to 22 dummy variables
df_temp = pd.get_dummies(df[['property_type']], prefix='pt_')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 11. room type, , processed to 4 dummy variables
df_temp = pd.get_dummies(df[['room_type']], prefix='rt_')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)

# 12. maximum number of people the property accommdates
df_select_0 = pd.concat([df_select_0, df['accommodates']], axis=1)

# 13. number of bathrooms avaiable
df_select_0 = pd.concat([df_select_0, df['bathrooms']], axis=1)

# 14. number of bedrooms available
df_select_0 = pd.concat([df_select_0, df['bedrooms']], axis=1)

# 15. number of beds available
df_select_0 = pd.concat([df_select_0, df['beds']], axis=1)

# 16. bed type, processed to 5 dummy variables
df_temp = pd.get_dummies(df[['bed_type']], prefix='bt_')
df_select_0 = pd.concat([df_select_0, df_temp], axis=1)
