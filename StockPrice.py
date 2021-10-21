
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