
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