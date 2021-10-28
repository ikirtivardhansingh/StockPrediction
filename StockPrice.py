
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