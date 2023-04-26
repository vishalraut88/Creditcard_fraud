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
            
        
        
    def missing_values_intable (self,df,threshold=100):
        missing_values = df.isnull().sum()
        missing_values_percentage = 100*df.isnull().sum()/len(df)
        missing_values_table = pd.concat([missing_values,missing_values_percentage],axis=1)
        missing_values_table = missing_values_table.rename(
          columns={0:'Missing Values',1:'Percentage of missing values'})
        missing_values_table = missing_values_table[missing_values_table['Percentage of missing values'] < threshold]
        return missing_values_table        
        