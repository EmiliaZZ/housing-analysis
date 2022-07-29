# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import os
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import datetime as dt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import json
import config
import pickle


global CURRENT_PATH
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
global pickles_path
pickles_path = os.path.join(CURRENT_PATH,"prepared_data")
os.chdir(CURRENT_PATH)

invest_ym = str(config.USER_PARAMS['invest_year'])+'.'+str(config.USER_PARAMS['invest_month']).zfill(2)

################  functions  ################
def load_and_clean_geojson(filename,df_final):
	with open(filename,'r') as jsonFile:
		data = json.load(jsonFile)
	temp=data
	# remove zipcodes not in df_final
	data_zips = []
	added_zips = []
	for i in range(len(temp['features'])):
		if ((temp['features'][i]['properties']['postalCode'] in list(df_final['zipcode'].unique()))&(temp['features'][i]['properties']['postalCode'] not in added_zips)):
			added_zips.append(temp['features'][i]['properties']['postalCode'])
			data_zips.append(temp['features'][i])
		
	zip_json = {}
	zip_json['type']='FeatureCollection'
	zip_json['features'] = data_zips
	return zip_json

################ import data  ################
# listings
listings = pd.read_pickle(os.path.join(pickles_path,"listings.pkl"))
listings.rename({'zip_clean':'zipcode','neighbourhood_group_cleansed':'neighborhood'}, axis=1, inplace=True)

# df final
df_final = pd.read_pickle(os.path.join(pickles_path,"df_final.pkl"))
df_final.rename({'count': 'listings_count'}, axis=1, inplace=True)

# forecasts
with open(os.path.join(pickles_path,"dict_forecasts.pickle"), 'rb') as handle:
	dict_forecasts = pickle.load(handle)

# geojson
geojson_file_name = config.PATHS['geojson_file_name']
zip_json = load_and_clean_geojson(os.path.join(CURRENT_PATH, geojson_file_name),df_final)
list_dropdown_values = [col for col in df_final if col !='zipcode']
list_dropdown_labels = []
for x in list_dropdown_values:
	x_new = x.replace("_", " ").capitalize().replace(' roi', ' ROI')
	list_dropdown_labels.append(x_new)
list_options = []
for i in range(0,len(list_dropdown_values)):
	dd = {'label':list_dropdown_labels[i],'value':list_dropdown_values[i]}
	list_options.append(dd)





##################### dash app ####################
# app = dash.Dash(__name__)


##############  layout  ##############
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = dbc.Card(
	[
		dbc.FormGroup(
			[
				dbc.Label("Select a metric to display: "),
				dcc.Dropdown(
					id='select_metric',
					options=list_options,
					value='payback_year'
					),
			]
		),
		dbc.FormGroup(
			[
				dbc.Label("Exclude zipcode with number of listings under:"),
				dcc.Input(
					id='exclude_zip_under',
					type='number',
					value=0
					),
			]
		)
	],
	body=True,
)

app.layout = dbc.Container(
	[
		html.H2("Return on investment analysis for 2-bedroom properties",style={'text-align': 'center'}),
		html.Br(),
		html.H5("Report is based on investments in "+invest_ym),
		html.Hr(),html.Br(),
		html.H5('Overall insights',style={'text-align': 'center'}),html.Br(),
		dbc.Row(
			[
				dbc.Col(controls, md=2),
				dbc.Col([
					html.Div(id='map_container', children=[],style={'text-align': 'center'}),
					dcc.Graph(id='display_choropleth_map')
					], md=5),
				dbc.Col([
					html.Div(id='scatter_container', children=[],style={'text-align': 'center'}),
					dcc.Graph(id='display_scatter'),
					html.Div(id='top_zips', children=[],style={'text-align': 'left'})
					], md=5),
			],
			align="center",
			no_gutters=True
		),
		html.Hr(),html.Br(),
		html.H5('Zip code level insights',style={'text-align': 'center'}),html.Br(),
		dbc.Row(
			[
				dbc.Col([
					html.Label('Input a Zip code'),
					dcc.Dropdown(
					id='select_zipcode',
					options= [{'label': x, 'value': x} for x in list(df_final['zipcode'].unique())],
					value=None
					)
					], md=2),
				dbc.Col([
					html.Div(id='hist_container', children=[],style={'text-align': 'center'}),
					dcc.Graph(id='display_hist')
					
					], md=5),
				dbc.Col([
					html.Div(id='forecast_container', children=[],style={'text-align': 'center'}),
					dcc.Graph(id='display_forecast')
					], md=5),
			],
			align="center",
			no_gutters=True
		),

	],
	fluid=True,
)


@app.callback(
	[Output(component_id='map_container', component_property='children'),
	 Output("display_choropleth_map", "figure")], 
	[Input("select_metric", "value"),
	Input("exclude_zip_under", "value")]
)
def display_choropleth(value,exclude_zip_under,df_final=df_final,list_options=list_options,zip_json=zip_json):
	df_map=df_final.copy()
	if exclude_zip_under!=None:
		df_map=df_map[df_map['listings_count']>=exclude_zip_under]  

	# get display label
	label = [list_options[i]['label'] for i in range(len(list_options)) if list_options[i]['value']== value][0]

	if value=='current_cost':
		df_map['display_text']=df_map.apply(lambda x: f"Zip code: {x.zipcode}<br>Median cost: {'${:,}'.format(int(x.current_cost)) }", axis=1)
	elif ('price' in value):
		df_map['display_text']=df_map.apply(lambda x: f"Zip code: {x.zipcode}<br>{label}: {'${:,.2f}'.format(round(x[value],2))}<br>Median cost: {'${:,}'.format(int(x.current_cost)) }", axis=1)
	elif (('roi' in value)|('rate' in value)):
		df_map['display_text']=df_map.apply(lambda x: f"Zip code: {x.zipcode}<br>{label}: {'{:}%'.format(round(x[value]*100,2))}<br>Median cost: {'${:,}'.format(int(x.current_cost)) }", axis=1)
	else: 
		df_map['display_text']=df_map.apply(lambda x: f"Zip code: {x.zipcode}<br>{label}: {round(x[value],2)}<br>Median cost: {'${:,}'.format(int(x.current_cost)) }", axis=1)
	
	# layout = go.Layout(height=700)
	container = f"{label} - map by zip code"
	fig = go.Figure(go.Choroplethmapbox(geojson=zip_json, locations=df_map['zipcode'], 
										z=df_map[value],featureidkey="properties.postalCode",
										colorscale="Spectral", marker_opacity=0.7, marker_line_width=1,
										text=df_map['display_text'],
										colorbar = dict(title = label),
										hoverinfo = "text",
									   ))
	fig.update_layout(mapbox_style="carto-positron",
					  mapbox_zoom=9.5,mapbox_center = {"lat":40.665 , "lon": -73.98} ,
					  margin={"r":0,"t":0,"l":0,"b":0}
					 )
	fig.update_traces(colorbar_thickness=20, selector=dict(type='choroplethmapbox'))
	return container, fig


### scatter plot callback
@app.callback(
	[Output(component_id='scatter_container', component_property='children'),
	 Output("display_scatter", "figure"),
	 Output(component_id='top_zips', component_property='children')], 
	[Input("select_metric", "value"),
	Input("exclude_zip_under", "value")]
)
def display_scatter(value,exclude_zip_under,df_final=df_final,list_options=list_options,listings=listings):
	df=df_final.copy()
	if exclude_zip_under!=None:
		df=df[df['listings_count']>=exclude_zip_under]  
	
	df1=listings[['zipcode','neighborhood']].drop_duplicates().reset_index(drop=True)
	df = pd.merge(df,df1,how='left',on=['zipcode'])
	# get display label
	label = [list_options[i]['label'] for i in range(len(list_options)) if list_options[i]['value']== value][0]
	
	if value=='current_cost':
		df['display_text']=df.apply(lambda x: f"Zip code: {x.zipcode}<br>Median cost: {'${:,}'.format(int(x.current_cost)) }", axis=1)
	elif ('price' in value):
		df['display_text']=df.apply(lambda x: f"Zip code: {x.zipcode}<br>{label}: {'${:,.2f}'.format(round(x[value],2))}<br>Median cost: {'${:,}'.format(int(x.current_cost)) }", axis=1)
	elif (('roi' in value)|('rate' in value)):
		df['display_text']=df.apply(lambda x: f"Zip code: {x.zipcode}<br>{label}: {'{:}%'.format(round(x[value]*100,2))}<br>Median cost: {'${:,}'.format(int(x.current_cost)) }", axis=1)
	else:
		df['display_text']=df.apply(lambda x: f"Zip code: {x.zipcode}<br>{label}: {round(x[value],2)}<br>Median cost: {'${:,}'.format(int(x.current_cost)) }", axis=1)
	
	# layout = go.Layout(height=700)
	container = f"{label} v.s. Median cost of properties by Zip code | size of dots: listings count"
	df_top = df.sort_values(['payback_year']).reset_index().head(3)
	msg_top_zips = f"3 Zip codes with shortest payback period: {', '.join(df_top['zipcode'].tolist())}"


	fig = px.scatter(df, x="current_cost", y=value, size='listings_count', color='neighborhood',
					 template='plotly_white',
					 custom_data=['display_text'],
					 labels={
					 value: label,
					 "current_cost": "Current median cost of property ($)"
					 }
					)
	fig.update_traces(textposition='top center',textfont_size=8,
					 hovertemplate="%{customdata[0]}"
					 )

	return container, fig, msg_top_zips



### histogram callback
@app.callback(
	[Output(component_id='hist_container', component_property='children'),
	 Output("display_hist", "figure")], 
	[Input("select_zipcode", "value")]
)
def display_hist(value,listings=listings):
	if (value!=None):
		df_hist=listings[listings['zipcode']==value].copy()
		container = f"Estimated nightly price distribution for 2-bedrooms in zip code {value} | Total listings: {'{:,}'.format(df_hist.shape[0])}"
	else:
		df_hist=listings.copy()
		container = f"Estimated nightly price distribution for 2-bedrooms in all zip codes | Total listings: {'{:,}'.format(df_hist.shape[0])}"

	fig = px.histogram(df_hist, x="nightly_price", 
		template='plotly_white',
		nbins=100,range_x=[0, 2000],
		color_discrete_sequence=['CadetBlue'],
		labels={'nightly_price':'Nightly price ($)','count':'Number of listings'}
		)


	return container, fig



### forecast callback
@app.callback(
	[Output(component_id='forecast_container', component_property='children'),
	 Output("display_forecast", "figure")], 
	[Input("select_zipcode", "value")]
)
def display_forecasts(value,dict_forecasts=dict_forecasts):
	if value !=None:
		zc_filter = value
		fig = plot_plotly(dict_forecasts[zc_filter]['ts_model'],dict_forecasts[zc_filter]['forecast'])
		container = f'Forecasted median property price for zip code {value}'
	else:
		container = 'Please select a Zip code to display time series forecast on median property price'
		fig={}
	
	return container, fig



if __name__ == '__main__':
	app.run_server(debug=True)