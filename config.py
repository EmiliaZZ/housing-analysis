USER_PARAMS = {
	'invest_year': 2021,
	'invest_month': 4,
	'get_roi_for':[5,10,15,20]

}

FORECAST_PARAMS = {
	'use_data_since_year': 2006,
	'use_data_since_month': 1,
	'uncertainty_interval': 0.9,
	'future_periods': 60,

	'estimate_revenue_for_year': 60,
	'inflation_rate': 0.02,
	'variable_cost_rate': 0.2,
	'multiplier_apply_year': [1],
	'multiplier': [0.5]

}

AIRBNB_DATA = {
	'download_url': "http://data.insideairbnb.com/united-states/ny/new-york-city/2019-07-08/data/listings.csv.gz",
	'col_needed': ['id','last_scraped','host_response_time','host_response_rate','neighbourhood_group_cleansed','city','state','zipcode','country','latitude','longitude','is_location_exact','property_type','room_type','bedrooms','square_feet','price','weekly_price','monthly_price','minimum_nights','maximum_nights','availability_30','availability_60','availability_90','availability_365','number_of_reviews','review_scores_rating','review_scores_location','reviews_per_month']

}

PATHS = {
	'geojson_file_name': 'nyc_zipcode_areas.geojson',

}

