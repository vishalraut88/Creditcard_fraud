import pandas as pd

class preprocessing:
	def __init__(self):
		pass

	def read_data(self,path):
		df = pd.read_csv(path)
		return df
		