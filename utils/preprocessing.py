import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class preprocessing:
    def __init__(self):
        pass

    def read_data(self,path):
        df = pd.read_csv(path)
        return df

    
    def scale_data(self,data,type_of_scaler=1):
        """
        1- Scale data using Standardscaler
        2- Scale data using MinMaxscaler
        """
        if(self.type_of_scaler==1):
            print("standard scaling")
            scaler=StandardScaler()
            data = scaler.fit_transform(data)
            return data
        if(self.type_of_scaler==2):
            print("Minmax scaling")
            scaler=MinmaxScaler()
            data = scaler.fit_transform(data)
            return data
            
        
        
        
        