# this class contains methods to predict the value of the stock price
class StockPrice(object):
	def __init__(self, company_name='GOOGL', steps=10, bs=128, dr=0.0, epochs=1000):
		self.steps = steps
		self.epochs = epochs
		self.batch_size = bs
		self.company_name = company_name
		self.dropout = dr
		self.load_data()
		self.create_subsets()
		self.build_model()
	# this function loads a model from a file
	def load_model_from_file(self, path):
		self.model = keras.models.load_model(path)
        # this function loads the data for a specific company
	def load_data(self):
		company = pickle.load(open(self.company_name+'.npy', 'r'))
		stock_prices = company.close.values.astype('float32')
		date_values = company.date.values
		dates = []
		cnt = 0
		for i in date_values:
			y, m, d = i.split('-')
			# ignore data older than 2007
			if int(y) < 2007:
				continue
			cnt += 1
			dates.append(m+'/'+d+'/'+y)
		self.dates = dates
		stock_prices = stock_prices.reshape(len(stock_prices), 1)
		stock_prices = stock_prices[len(stock_prices)-cnt:]
			# scale the values
		self.scaler = MinMaxScaler(feature_range=(0, 1))
		self.stock_prices = self.scaler.fit_transform(stock_prices)
	# this function splits the data into training and testing datasets for cross validation
	def create_subsets(self, tr_percentage=0.7):
		# split data into training set and test set
		l = len(self.stock_prices)
		train_size = int(l * tr_percentage)
		train = self.stock_prices[0:train_size, :]
		test_size = l - train_size
		test = self.stock_prices[train_size:l, :]
		print 'Training set size = ', len(train)
		print 'Testing set size = ', len(test)
		# convert stock prices into time series dataset
		trainX, self.trainY = create_dataset(train, self.steps)
		testX, self.testY = create_dataset(test, self.steps)
		# reshape input of the LSTM to be format [samples, time steps, features]
		self.trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
		self.testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	# this function designs the main NN structure to be trained
	def build_model(self):
		self.model = Sequential()
		# Model M1
		self.model.add(LSTM(20, input_shape=(self.steps, 1)))
		self.model.add(Dense(1))
		self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
		# this function trains the current NN model
	def train(self):
		self.model.fit(self.trainX, self.trainY, epochs=self.epochs,
			batch_size=self.batch_size)
		self.model.save('./'+self.company_name+'_model-bs'+str(self.batch_size)+'-steps'+str(self.steps))
	# this function evaluates the performance of the trained model on both the training and testing datasets
	def evaluate(self):
		# make predictions
		trainPredict = self.model.predict(self.trainX)
		testPredict = self.model.predict(self.testX)
		# invert predictions and targets to unscaled
		self.trainPredict = self.scaler.inverse_transform(trainPredict)
		self.trainY = self.scaler.inverse_transform([self.trainY])
		self.testPredict = self.scaler.inverse_transform(testPredict)
		self.testY = self.scaler.inverse_transform([self.testY])
		# calculate errors
		self.trainRMSE = math.sqrt(mean_squared_error(self.trainY[0], self.trainPredict[:,0]))
		self.trainMAPE = np.mean(np.abs((self.trainY[0] - self.trainPredict[:,0]) / self.trainY[0])) * 100
		print 'Train Score: RMSE =', self.trainRMSE, ' MAPE =', self.trainMAPE
		self.testRMSE = math.sqrt(mean_squared_error(self.testY[0], self.testPredict[:,0]))
		self.testMAPE = np.mean(np.abs((self.testY[0] - self.testPredict[:,0]) / self.testY[0])) * 100
		print 'Test Score: RMSE =', self.testRMSE, ' MAPE =', self.testMAPE
	# this function plots the results
	def plot_results(self):
		import matplotlib.pyplot as plt
		import matplotlib.dates as mdates
		import datetime as dt
		d = [dt.datetime.strptime(i, '%m/%d/%Y').date() for i in self.dates]
		# shift predictions of training data for plotting
		trainPredictPlot = np.empty_like(self.stock_prices)
		trainPredictPlot[:, :] = np.nan
		trainPredictPlot[self.steps:len(self.trainPredict)+self.steps, :] = self.trainPredict
		# shift predictions of test data for plotting
		testPredictPlot = np.empty_like(self.stock_prices)
		testPredictPlot[:, :] = np.nan
		testPredictPlot[len(self.trainPredict)+(self.steps*2)+1:len(self.stock_prices)-1, :] = self.testPredict
		fig, ax = plt.subplots() # create a new figure with a default 111 subplot
		from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
		from mpl_toolkits.axes_grid1.inset_locator import mark_inset
		plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
		plt.gca().xaxis.set_major_locator(mdates.YearLocator())
		ax.plot(d, self.scaler.inverse_transform(self.stock_prices), 'k', label='True')
		ax.plot(d, trainPredictPlot, 'xkcd:royal blue', label='Training Prediction')
		ax.plot(d, testPredictPlot, 'xkcd:cerulean', label='Testing Prediction')
		legend = plt.legend(loc='upper left', shadow=False, fontsize='large')
		plt.title('Goole Stock Price', fontsize=16)
		plt.xlabel('Time', fontsize=14)
		plt.ylabel('Value', fontsize=14)
		plt.gcf().autofmt_xdate()
		from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
		axins = zoomed_inset_axes(ax, 23.0, loc=4) # zoom-factor: 2.5, location: upper-left
		axins.plot(d, self.scaler.inverse_transform(self.stock_prices), 'k', label='True')
		axins.plot(d, trainPredictPlot, 'xkcd:royal blue', label='Training Prediction')
		axins.plot(d, testPredictPlot, 'xkcd:cerulean', label='Testing Prediction')
		x1 = dt.date(2015, 11, 14)
		x2 = dt.date(2015, 12, 5)
		axins.set_xlim(x1, x2)
		axins.set_ylim(760, 785)
		plt.yticks(visible=False)
		plt.xticks(visible=False)
		from mpl_toolkits.axes_grid1.inset_locator import mark_inset
		mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
		plt.show()
	# this function loads a model from a file
	def load_model_from_file(self, path):
		self.model = keras.models.load_model(path)
	# this function saves the model to a file
	def save_model_to_file(self):
		from keras.utils import plot_model
		plot_model(self.model, show_layer_names=False, to_file='model1-google.eps')
