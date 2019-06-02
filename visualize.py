import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
from numpy.polynomial.polynomial import polyfit


def visualize(bikeDF, savedir):
	"""Function to generate and save a set of visualizations
	for the bike dataset.

	Args:
		bikeDF: cleaned dataframe for bike dataset
		savedir: location where visualizations will be saved.

	Returns:
		None
	"""
	if not os.path.exists(savedir):
		os.mkdir(savedir)

	# distribution of count variable
	fig, ax = plt.subplots(figsize=(20,10))
	sn.distplot(bikeDF['cnt'], ax=ax)
	plt.savefig(savedir + 'cnt_distribution.png')

	# season-wise total bike count
	plt.figure()
	plt.bar(x=bikeDF['season'],height=bikeDF['cnt'])
	plt.title("Bike usage by season")
	plt.xticks(range(1,5), ['spring','summer','fall','winter'], rotation=60)
	plt.xlabel('Season')
	plt.ylabel("Bike count")
	plt.savefig(savedir + 'season-cnt.png')

	# daily bike usage count
	plt.figure()
	plt.bar(x=bikeDF['weekday'],height=bikeDF['cnt'])
	plt.title("Bike usage by day of the week")
	plt.xticks(np.arange(7), ['0','1','2','3','4','5','6'], rotation=60)
	plt.xlabel("Day")
	plt.ylabel("Bike count")
	plt.savefig(savedir + 'daily-cnt.png')

	# monthly bike count for both years
	fig, ax = plt.subplots(figsize=(20,10))
	sn.pointplot(data=bikeDF[['mnth', 'cnt', 'yr']],x='mnth', y='cnt', hue='yr', ax=ax)
	ax.set(title="Bike usage count over the months for two years")
	plt.savefig(savedir + 'month-cnt.png')

	# hourly bike usage count for all users,registered users and casual users
	for feature, name in {'cnt':'all','registered':'registered','casual':'casual'}.items():
		fig, ax = plt.subplots(figsize=(20,10))
		sn.pointplot(data=bikeDF[['hr',feature,'weekday']], x='hr', y=feature,hue='weekday', ax=ax)
		ax.set(title="Bike count during the day by weekday for {} users".format(name))
		plt.savefig(savedir + name + '-count.png')

	# hourly bike usage count by season
	fig, ax = plt.subplots(figsize=(20,10))
	sn.pointplot(data=bikeDF[['hr','cnt','season']], x='hr', y='cnt',hue='season', ax=ax)
	ax.set(title="Bike usage count during the day by season")
	plt.savefig(savedir + 'season-hr-count.png')

	# hourly bike usage count by month
	fig, ax = plt.subplots(figsize=(20,10))
	sn.pointplot(data=bikeDF[['hr', 'cnt', 'mnth']],x='hr', y='cnt', hue='mnth', ax=ax)
	ax.set(title="Bike usage count during the day by month")
	plt.savefig(savedir + 'month-hr-count.png')

	# hourly bike usage count by weather
	fig, ax = plt.subplots(figsize=(20,10))
	sn.pointplot(data=bikeDF[['hr', 'cnt', 'weathersit']],x='hr', y='cnt', hue='weathersit', ax=ax)
	ax.set(title="Bike usage count during the day by weather")
	plt.savefig(savedir + 'weather-hr-count.png')

	# scatter plots for hourly bike usage count by hum, temp and windspeed
	# fit a line to the scatter plots show trend
	for feature in ['hum','temp','windspeed']:
		x=bikeDF[feature]
		y=bikeDF['cnt']
		plt.figure()
		plt.scatter(x,y)
		plt.xlabel(feature)
		plt.ylabel('hour count')
		plt.title("Hourly bike usage vs. {}".format(feature))
		b, m = polyfit(x, y, 1)
		plt.plot(x,b+m*x,'-',color='red')
		plt.savefig(savedir + feature + '-hr-count.png')

	# box plots for visualizing the outliers
	fig, ax = plt.subplots(figsize=(10, 5))
	sn.boxplot(x=bikeDF['hr'], y=bikeDF['cnt'], ax=ax)
	ax.set(title="Boxplot for hour of the day")
	plt.savefig(savedir + 'outliers.png')

	# correlation coefficient among continuous variables
	matrix = bikeDF[['temp','atemp','hum','windspeed', 'cnt']].corr()
	heat = np.array(matrix)
	heat[np.tril_indices_from(heat)] = False
	fig, ax = plt.subplots()
	fig.set_size_inches(20, 10)
	sn.heatmap(matrix, mask=heat, vmax=1.0, vmin=0.0, square=True, annot=True, cmap="Reds")
	plt.savefig(savedir + 'correlation.png')