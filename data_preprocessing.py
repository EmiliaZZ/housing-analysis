import os
import numpy as np
import pandas as pd
import re
import requests
import datetime as dt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import config

######################################## Zillow data ########################################

## subset raw cost dataframe to inputed city and only keep zipcode & historic price columns
def city_subset_clean(df,city,state):
	'''
	ACCEPTS - df: raw cost data, city: string filter for city, state: string filter for state
	PERFORMS - clean, and subset raw cost data with only desired city
	RETURNS -  cleaned df
	'''
	# convert zipcode to string and add leading 0 if needed
	df['RegionName'] = df['RegionName'].astype(str).str.zfill(5)
	df1 = df[(df['City']==city) & (df['State']==state)].reset_index(drop=True).copy()
	df1.rename({'RegionName': 'zipcode'}, axis=1, inplace=True)
	## remove all non-price col except zipcode
	col_to_rm = [col for col in df1.columns if ((isinstance(col, dt.date)==False)&(col!='zipcode'))]
	df_clean = df1.drop(col_to_rm, axis = 1)
	num_zips = len(df_clean.zipcode.unique())
	msg = f"{str(num_zips)} zipcodes are found in {city}, {state}"

	return df_clean, msg


## forecast 
def forecast_cost_for_all_zip(df, use_data_since, uncertainty_interval=0.9, future_periods=60):
	'''
	ACCEPTS - df: cleaned cost data, use_data_since: datetime variable - earliest date that the model uses to predict, future_periods: int - months to predict
	PERFORMS - forecast using fbprophet
	RETURNS -  dictionary of all forecasts
	'''
	# remove price columns before the starting date
	col_to_rm = []
	for col in df.columns:
		if (isinstance(col, dt.date)==True):
			if (col<use_data_since):
				col_to_rm.append(col)     
	df_clean = df.drop(col_to_rm, axis = 1)
	## initiate dictionary for all zipcodes
	dict_all = {}

	## interate for each zip
	for zc in df_clean['zipcode'].unique():
		df_zip = df_clean[df_clean['zipcode']==zc].dropna(axis=1, how='all').reset_index()
		df_zip.drop(['zipcode'], axis=1, inplace=True)
		df_zip_t = df_zip.T.reset_index().iloc[1:]
		df_zip_t.columns = ['ds','y']
		df_zip_t['ds'] = pd.DatetimeIndex(df_zip_t['ds'])

		# fit prophet model
		ts_model = Prophet(interval_width=uncertainty_interval).fit(df_zip_t)
		future_dates = ts_model.make_future_dataframe(periods=future_periods, freq='MS')
		forecast = ts_model.predict(future_dates) 
		dict_zip = {'ts_model':ts_model, 'forecast':forecast}
		dict_all[zc] = dict_zip
		
	return dict_all


######################################## Airbnb data ########################################

def download_airbnb_data(url, col_needed):
	'''
	ACCEPTS - url: string url, col_needed: list of column names to import
	PERFORMS - load airbnb data
	RETURNS -  dataframe
	'''
	filename = url.split("/")[-1]
	with open(filename, "wb") as f:
		r = requests.get(url)
		f.write(r.content)

	df = pd.read_csv(filename, 
							  compression='gzip',
							  header=0,
							  usecols=col_needed)
	return df


def clean_airbnb_2b(df, state_str_match, zip_filter=None, exclude_size_below=None):
	'''
	ACCEPTS - df: airbnb listings df, state_str_match: string values to filter by (case insensitive), zip_filter: list of zipcodes, exclude_size_below: int
	PERFORMS - clean up zipcode -> 'zip_clean' column & calculate nightly price for 2b listings -> 'nightly_price' column
	RETURNS -  cleaned df
	'''
	df['zipcode']=df['zipcode'].apply(str)
	df['state']=df['state'].apply(str)
	# remove rows with null zipcode & bad states
	df1 = df[(df['zipcode'].notna()) & (df['state'].str.match(state_str_match,case=False)==True)]
	# only take first 5 digits of zipcode
	df1['zip_clean'] = df1['zipcode'].str.strip().str[0:5]
	# remove zipcodes with characters other than digits 
	df1 = df1[~df1['zip_clean'].str.contains(r'[^0-9]')].reset_index(drop=True)
	# reassign neighborhood to mode
	df_temp = df1.groupby(['zip_clean','neighbourhood_group_cleansed'], as_index=False).size().sort_values(['zip_clean','size'],ascending=(True,False)).groupby(['zip_clean']).first().reset_index()
	df1 = pd.merge(df1.loc[:, df1.columns != 'neighbourhood_group_cleansed'],df_temp,how='right',on=['zip_clean'])
	num_zips_df1 = len(df1['zip_clean'].unique())
	# only keep zipcodes that we have cost data for - zip filter
	if zip_filter!=None:
		df1 = df1[df1['zip_clean'].isin(zip_filter)].reset_index(drop=True)
	# only keep zipcodes that have more than n listings
	if exclude_size_below!=None:
		df_temp = df1['zip_clean'].value_counts().reset_index()
		small_zips = list(df[df['zip_clean']<exclude_size_below]['index'])
		df1 = df1[~df1['zip_clean'].isin(small_zips)].reset_index(drop=True)
	# add nightly price for 2b
	df2 = df1[((df1['bedrooms']==1)&(df1['room_type']=='Private room')) | ((df1['bedrooms']==2)&(df1['room_type']=='Entire home/apt'))].reset_index(drop=True)
	df2['price'] = df2['price'].map(lambda x: float(re.sub(r'[^\d.]', '', x)) if type(x)==str else float(x))
	df2['nightly_price'] = np.where(df2['bedrooms']==2, df2['price'], df2['price']*2)

	num_zips_df2 = len(df2['zip_clean'].unique())
	msg = f"{num_zips_df1} zipcodes are found in {state_str_match}, among these {num_zips_df2} also have cost data -> {num_zips_df2} are kept for later analysis"

	return df2, msg



def boxplot_by_zip(df,var,var_display):
	'''
	ACCEPTS - df: airbnb listings df, var: string var name to plot, var_display: stirng var name to display
	PERFORMS - clean up zipcode -> 'zip_clean' column & calculate nightly price for 2b listings -> 'nightly_price' column
	RETURNS -  cleaned df
	'''
	grouped = df.groupby(['zip_clean']).agg({var:'median'}).sort_values(by=[var], ascending=False)
	df_plot = df[['zip_clean',var,'neighbourhood_group_cleansed']]
	df_plot['count'] = df_plot.groupby('zip_clean')[var].transform('count').astype('str')+' listings'
	max_y = df_plot[var].mean()+4*df_plot[var].std()
	p = px.box(df_plot,x='zip_clean',y=var,
		color = 'neighbourhood_group_cleansed',
		category_orders={'zip_clean':grouped.index},
		range_y=[0,max_y],
		boxmode='overlay',
		title=f'{var_display} of 2B properties by Zipcode, ordered by median',
		hover_name='count',
		template = 'plotly_white',
		labels={"zip_clean": "Zipcode",var: var_display,"neighbourhood_group_cleansed":"Neighborhood" },
		color_discrete_sequence=px.colors.qualitative.T10    
		)

	return p


def get_airbnb_rev_by_zip(df,dict_forecasts,invest_yy=2021,invest_m=4):
	'''
	ACCEPTS - df: cleaned airbnb listings df, dictionary of fbprophet forecasts, inflation rate: float
	PERFORMS - aggregate nightly price & occupancy rate for zipcode and calculate current nightly price
	RETURNS -  zipcode level df
	'''
	df1 = df.groupby('zip_clean',as_index=False).agg({'id':'count','nightly_price':['median','mean','std'],'availability_30':['median','mean','std']})
	df1.columns = ['zipcode','count','median_nightly_price','mean_nightly_price','std_nightly_price','median_avail_30','mean_avail_30','std_avail_30']
	df1['est_occupancy_rate'] = (30-df1['mean_avail_30']*0.7)/30

	list_appreciation = []
	for zc in dict_forecasts.keys():
		str_zc = str(zc)
		df_zip_tail = dict_forecasts[zc]['forecast'].sort_values('ds').tail(36).reset_index()
		# estimated annual appreciation rate: average annual appreciation rate in the latest 3 years of forecasted data
		est_annual_apprn_rate = (df_zip_tail.loc[35,'yhat']/df_zip_tail.loc[0,'yhat'])**(1./3.)-1
		d_zip = {'zipcode':str_zc, 'est_annual_apprn_rate':est_annual_apprn_rate}
		list_appreciation.append(d_zip)
		
	df_appreciation = pd.DataFrame(list_appreciation)
	df1 = pd.merge(df1,df_appreciation,on=['zipcode'])

	# compute current nightly price
	invest_dt=dt.datetime(invest_yy, invest_m, 1, 0, 0)
	last_scraped_dt=dt.datetime.strptime(df.last_scraped[0], '%Y-%m-%d')
	scraped_to_invest = (invest_dt.year - last_scraped_dt.year) * 12 + invest_dt.month - last_scraped_dt.month
	df1['current_price'] = df1['median_nightly_price']*((1+df1['est_annual_apprn_rate'])**((scraped_to_invest//12)+ (scraped_to_invest%12)/12))
	
	return df1



def est_future_gross_income(df,est_for_yr=50,inflation_rate=0.02,variable_cost_rate=0.2,multiplier_apply_year=[1],multiplier=[0.5]):
	'''
	ACCEPTS - df: by_zip df, est_for_yr: int, # of year to forecast, inflation_rate: float, variable_cost_rate: float, multiplier_apply_year: list of year to apply multipler,multiplier: list of multipler
	PERFORMS - aggregate nightly price & occupancy rate for zipcode and calculate current nightly price
	RETURNS -  zipcode level df
	'''
	df_est_gross_income = pd.DataFrame(columns=['year','est_annual_gross_income','zipcode'])
	for zc in df['zipcode'].unique():
		df_zc = df[df['zipcode']==zc].reset_index(drop=True)
		real_apprn_rate = max(0,(df_zc['est_annual_apprn_rate'][0]-inflation_rate)) # real appraciation rate = estimated appreciation rate - inflation rate
		list_yy = list(range(1,(est_for_yr+1)))
		list_est_annual_income = [(df_zc['current_price'][0]*(1+real_apprn_rate)**yy)*(1-variable_cost_rate)*365*(df_zc['est_occupancy_rate'][0]) for yy in list_yy]
		df_zc_future_income = pd.DataFrame({'year':list_yy,'est_annual_gross_income':list_est_annual_income})
		df_zc_future_income['zipcode'] = zc
		df_est_gross_income = df_est_gross_income.append(df_zc_future_income, ignore_index=True)
	## apply custom multipliers to annual gross income for certain years
	dict_multipliers = dict(zip(multiplier_apply_year, multiplier))
	for key, value in dict_multipliers.items():
		df_est_gross_income.loc[df_est_gross_income['year']==key, 'est_annual_gross_income'] = df_est_gross_income['est_annual_gross_income']*value

	df_est_gross_income['cum_income'] = df_est_gross_income.groupby('zipcode',as_index=False)['est_annual_gross_income'].cumsum()

	return df_est_gross_income


def get_roi(dict_forecasts,df_est_gross_inc,df_zip_rev,invest_yy = 2021,invest_m = 4,get_roi_for_year = [5,10,15,20]):
	'''
	ACCEPTS - phrophet forecast output, estimated income df, revenue by zip df, invest_yy: int,invest_m: int, get_roi_for_year: list of int
	PERFORMS - create final dataframe with payback period and ROI
	RETURNS -  dataframe
	'''

	## get currenct cost from ts forecasted results
	list_zips = []
	list_current_cost = []
	for zc in dict_forecasts.keys():
		str_zc = str(zc)
		df = dict_forecasts[zc]['forecast']
		current_cost = df[(df['ds'].dt.year==invest_yy) & (df['ds'].dt.month==invest_m)].reset_index(drop=True).loc[0,'yhat']
		list_zips.append(str_zc)
		list_current_cost.append(current_cost)
	df_all = pd.DataFrame({'zipcode': list_zips,'current_cost': list_current_cost})
	# calculate payback period in years
	df_all = pd.merge(df_all,df_est_gross_inc,on=['zipcode'])
	df_all = df_all[df_all['cum_income']>=df_all['current_cost']].groupby('zipcode',as_index=False).first().reset_index(drop=True)
	df_all['payback_year'] = df_all['year']-((df_all['cum_income']-df_all['current_cost'])/df_all['est_annual_gross_income'])
	df_all.drop(['year','est_annual_gross_income','cum_income'],axis=1,inplace=True)

	# get roi 
	for y in get_roi_for_year:
		if y<=df_est_gross_inc['year'].max():
			colname = f'{y}y_roi'
			df_income = df_est_gross_inc[df_est_gross_inc['year']==y].groupby(['zipcode'],as_index=False)['cum_income'].first()
			
			df_all=pd.merge(df_all,df_income,on=['zipcode'])
			df_all[colname] = df_all['cum_income']/df_all['current_cost']
			df_all.drop(['cum_income'],axis=1,inplace=True)
	# add other info
	df_all=pd.merge(df_all,df_zip_rev[['zipcode','count','est_occupancy_rate','std_nightly_price','est_annual_apprn_rate']],on=['zipcode'])

	return df_all