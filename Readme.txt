===================================================
Bike Sharing Dataset: EDA and Predictive Modeling
===================================================

This project contains files for visualizing the bike dataset, training and testing a predictive model on it and predicting hourly bike usage count 'cnt' values using a saved model.


=========================================
Files
=========================================

	- Readme.txt
	- hour.csv : bike sharing counts aggregated on hourly basis. Records: 17379 hours
	- bike_eda.py: main script to generate visualizations from data and do other data handling tasks
        - vusualize.py: contains code for the visualize function
        - model.py: main script to train, test and predict from a saved model 

	
=========================================
Dataset characteristics
=========================================	
Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
	
	- instant: record index
	- dteday : date
	- season : season (1:spring, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2011, 1:2012)
	- mnth : month ( 1 to 12)
	- hr : hour (0 to 23)
	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	- hum: Normalized humidity. The values are divided to 100 (max)
	- windspeed: Normalized wind speed. The values are divided to 67 (max)
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered
	
=========================================
How to run:
=========================================
1. To generate visualizations and explore dataset:
        python bike_eda.py --datafile='hour.csv' --save_plot='Plots/'

2. To train model, get R-square performance metric and to predict using a saved trained model:
    Train:
        python bike_test.py --datafile='hour.csv' --mode='train' --model_path='model/model.sav' 
    Predict:
        python bike_test.py --datafile='hour.csv' --mode='predict' --model_path='model/model.sav' 

