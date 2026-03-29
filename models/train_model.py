import tensorflow as tf
import numpy as np
import pandas as pd

import datetime
def mse(true:np.array, pred:np.array):
    """
    Function for evaluation MSE score
    """
    resid= tf.reduce_mean(tf.square(true- pred))
    return resid
def mae(true:np.array, pred:np.array):
    """
    Function for evaluation MAE score
    """
    resid= tf.reduce_mean(tf.abs(true - pred))
    return resid

def train_model(model:tf.keras.Model,train_data_x:np.array,train_data_y:np.array,output_path:str):
    """
    Function for training keras model on data.
    model:tf.keras.Model - model for training
    train_data_x:np.array - samples for training (x)
    train_data_y:np.array - real data for training (y)
    output_path:str - path for saved model files
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mae'
    )
    model.fit(
        train_data_x, # input
        train_data_y, # equals output
        validation_split=0.2, # prevent overfitting
        epochs=1000,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5,monitor="val_loss",restore_best_weights=True) # early stop
        ]
    )
    
    model.encoder.save(f"{output_path}/{model.name}_encoder.keras")
    model.decoder.save(f"{output_path}/{model.name}_decoder.keras")
    return model

def test_model(model:tf.keras.Model,test_data_x:np.array,test_data_y:np.array):
    """
    Function for testing keras model on data.
    model:tf.keras.Model - model for training
    test_data_x:np.array - samples for testing (x)
    test_data_y:np.array - real data for testing (y)
    """
    results={"MAE":None,"MSE":None}
    y_pred=model.predict(test_data_x)
    results["MAE"]=float(mae(test_data_y,y_pred))
    results["MSE"]=float(mse(test_data_y,y_pred))
    return results

def permutation_feature_importance(model:tf.keras.Model,test_x:np.array,test_y:np.array,n_permutations:int=1):
    """
    Function for permutation feature importance
    model:tf.keras.Model - model for training
    test_data_x:np.array - samples for testing (x)
    test_data_y:np.array - real data for testing (y)
    n_permutations:int - number of permutations on one feature
    """
    feature_importances = np.zeros((test_x.shape[-1],test_x.shape[-1]))
    for feature_index1 in range(len(feature_importances)):
        print('Running permutations for feature:', feature_index1)
        for _ in range(n_permutations):
            shuffled_test_x = test_x.copy()
            if len(shuffled_test_x.shape)>2:
                np.random.shuffle(shuffled_test_x[:, :,feature_index1])
            else:
                np.random.shuffle(shuffled_test_x[:, feature_index1])
            y_pred=model.predict(shuffled_test_x)
            print(y_pred.shape)
            for feature_index2 in range(len(feature_importances)):
                shuffled_loss = float(tf.reduce_mean(tf.abs(test_y[:,feature_index2] - y_pred[:,feature_index2])))
                feature_importances[feature_index1,feature_index2] = shuffled_loss
    return feature_importances


def monthes_permutation_feature_importance(model:tf.keras.Model,dataset:pd.DataFrame,low_time_border:datetime.datetime,high_time_border:datetime.datetime,time_length:int=None):
    """
    Function for monthes permutation feature importance
    dataset:pd.Dataframe - dataset
    """
    all_data=np.array(dataset.drop("datetime",axis=1))
    curr_month=None
    if time_length==None:    
        time_data=[]
        month_data=[]
        for ind,dt in enumerate(dataset["datetime"]):
            if dt.date()<low_time_border:
                continue
            if len(month_data)==0 and len(time_data)==0:
                curr_month=dt.month
            if dt.month!=curr_month:
                curr_month=dt.month
                time_data.append(np.array(month_data.copy()))
                month_data=[]
            month_data.append(all_data[ind].copy())
            if dt.date()>=high_time_border:
                break
        time_data.append(np.array(month_data.copy()))
        fi_for_months=[]
        for ind,month_data in enumerate(time_data):
            print(ind)
            fi=permutation_feature_importance(model,month_data,month_data,1)
            fi=np.abs(fi)
            fi_for_months.append(fi.copy())
        fi_for_months=np.array(fi_for_months)
        return fi_for_months
    else:
        time_data=[]
        month_data=[]
        for ind,dt in enumerate(dataset["datetime"]):
            if dt.date()<low_time_border:
                continue
            if len(month_data)==0 and len(time_data)==0:
                curr_month=dt.month
            if dt.month!=curr_month:
                curr_month=dt.month
                time_data.append(np.array(month_data.copy()))
                month_data=[]
            month_data.append(all_data[ind].copy())
            if dt.date()>=high_time_border:
                break
        if len(month_data.copy())!=0:
            time_data.append(np.array(month_data.copy()))
        print(len(time_data))
        time_data_x=[]
        time_data_y=[]
        for ind,month_data in enumerate(time_data):
            print(ind)
            if ind>11:
                break
            time_data_x.append(np.array([month_data[i:time_length+i] for i in range(month_data.shape[0]-time_length)]))
            time_data_y.append(np.array([month_data[time_length+i] for i in range(month_data.shape[0]-time_length)]))
        fi_for_months=[]
        for ind in range(len(time_data_x)):
            fi=permutation_feature_importance(model,time_data_x[ind],time_data_y[ind],1)
            fi=np.abs(fi)
            fi_for_months.append(fi.copy())
        
        fi_for_months=np.array(fi_for_months)
        return fi_for_months
        
    
        
    