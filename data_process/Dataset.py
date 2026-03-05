
import pandas as pd
import numpy as np
class Dataset:
    def __init__(self,dataset_path:str=None,coords_path:str=None):
        if dataset_path!=None:
            self.load_dataset_from_file(dataset_path)
        
        if coords_path!=None:
            self.load_coords_from_file(coords_path)
    
    def load_dataset_from_file(self,path:str):
        if path.split(".")[-1]=="csv":
            self.load_dataset_from_csv_file(path)
        else:
            raise Exception("Cant process not csv files yet")
    
    def load_coords_from_file(self,path:str):
        if path.split(".")[-1]=="csv":
            self.load_coords_from_csv_file(path)
        else:
            raise Exception("Cant process not csv files yet")
    
    def load_dataset_from_csv_file(self,path:str):
        data=pd.read_csv(path)
        if "datetime" not in data.columns:
            raise Exception("Column 'datetime' not found in dataset")
        data["datetime"]=pd.to_datetime(data["datetime"])
        self.data=data
        return self.data
    def load_coords_from_csv_file(self,path:str):
        coords=pd.read_csv(path)
        if "site" not in coords.columns:
            raise Exception("Column 'site' not found in dataset")
        if "lat" not in coords.columns:
            raise Exception("Column 'lat' not found in dataset")
        if "lon" not in coords.columns:
            raise Exception("Column 'lon' not found in dataset")
        
        self.coords=coords
        return self.coords
    
    def make_base_dataset(self):
        if self.data is None:
            raise Exception("Data not loaded")
        
        raw_data=self.data.copy()
        for column in raw_data.columns:
            if column!="datetime":
                raw_data[column]=raw_data[column]/max(raw_data[column].max(),1)
        
        all_data=np.array(raw_data.drop("datetime",axis=1))
        train_data=all_data[:int(all_data.shape[0]*0.8)]
        test_data=all_data[int(all_data.shape[0]*0.8):]
        return {"test_data":test_data,
                "train_data":train_data}
    
    def make_time_dataset(self,time_length:int):
        if self.data is None:
            raise Exception("Data not loaded")
        
        raw_data=self.data.copy()
        for column in raw_data.columns:
            if column!="datetime":
                raw_data[column]=raw_data[column]/max(raw_data[column].max(),1)
        
        all_data=np.array(raw_data.drop("datetime",axis=1))
        X=np.array([all_data[i:time_length+i] for i in range(all_data.shape[0]-time_length)])
        Y=np.array([all_data[time_length+i] for i in range(all_data.shape[0]-time_length)])
        train_data_x=X[:int(X.shape[0]*0.8)]
        train_data_y=Y[:int(Y.shape[0]*0.8)]
        test_data_x=X[int(X.shape[0]*0.8):]
        test_data_y=Y[int(Y.shape[0]*0.8):]
        return {"test_data_x":test_data_x,
                "test_data_y":test_data_y,
                "train_data_x":train_data_x,
                "train_data_y":train_data_y}
    


    