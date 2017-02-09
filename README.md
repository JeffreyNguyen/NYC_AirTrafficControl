# NYC_AirTrafficControl

The code here can be used to predict air pollution (PM2.5) levels given a variety usage of different heating sources, the amount of traffic on the road, and the weather. 

The code uses a variety of metrics to build an autoregressive model to predict the a certain days PM2.5 levels given the weather, pollution, and traffic from the previous days. The model runs both multilinear LASSO regression and artificial neural networks to predict the air pollution. The prediction quality has R^2=0.5 and a mean absolute error of ~4.5. The program also runs simulations for reduced traffic situations in different locations in order to suggest where to limit traffic in order to best improve air quality.

The program uses multiple datasets saved as CSV files. They are :
	PM2.5 measurements from around the city (air*.csv), at 15 locations
	Weather data, including rain, temperature, wind, wind direction, humidity, and others (nyc_weather.csv)
	Heating Oil and Gas usage (Heating_Oil_Consumption.csv and Heating_Gas_Consumption.csv)
	Measurements of traffic density from different locations.  from around the city (traffic_summary.csv)
	
The data for this site can be easily visualized at www.jeffreynguyen.top

## Getting Started

To load and combine all data sets into a pandas dataframe, run

```
from weather import *
d_origin=load_data_all()
```

This will allow you to explore the datasets. To load data and run simulations, use 
```
run.py
```

