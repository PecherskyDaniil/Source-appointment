import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
import datetime
from data_process.choose_file import choose_files
from data_process.Dataset import Dataset
from models.train_model import train_model,test_model,permutation_feature_importance,monthes_permutation_feature_importance
from models.SeqAutoencoder import SeqAutoencoder
from models.ConvAutoencoder import ConvAutoencoder
from models.CompressedConvAutoencoder import CompressedConvAutoencoder
from models.LSTMAutoencoder import LSTMAutoencoder
from viz.charts import create_feature_importance_chart,create_feature_importance_map,draw_month_maps
# Загрузка данных из файла
dataset_path,coords_path=choose_files()
main_dataset=Dataset(dataset_path,coords_path)
output_path="./output"
in_out_dim=main_dataset.data.columns.size-1
bottleneck=int(in_out_dim/2.5)


print(f"{in_out_dim} входных данных,{bottleneck} - бутылочное горлышко между энкодером и декодером")
# Подготовка данных для полносвязной модели
for model_name in ["sequential_model","convolutional_model","lstm_model"]:
    if model_name=="sequential_model":
        time_length=None
        dataset=main_dataset.make_base_dataset()
        train_data_x=dataset["train_data"]
        train_data_y=dataset["train_data"]
        test_data_x=dataset["test_data"]
        test_data_y=dataset["test_data"]
    else:
        if in_out_dim>150:
            time_length=100
        else:
            time_length=180
        dataset=main_dataset.make_time_dataset(time_length)
        train_data_x=dataset["train_data_x"]
        train_data_y=dataset["train_data_y"]
        test_data_x=dataset["test_data_x"]
        test_data_y=dataset["test_data_y"]

    match model_name:
        case "sequential_model":
            model=SeqAutoencoder(name=model_name,in_out_dim=in_out_dim,bottleneck_dim=bottleneck)
        case "convolutional_model":
            if in_out_dim>150:
                model=CompressedConvAutoencoder(name=model_name,in_shape=(time_length,in_out_dim,1),bottleneck_dim=bottleneck,out_dim=in_out_dim)
            else:
                model=ConvAutoencoder(name=model_name,in_shape=(time_length,in_out_dim,1),bottleneck_dim=bottleneck,out_dim=in_out_dim)
        case "lstm_model":
                model=LSTMAutoencoder(name=model_name,in_shape=(time_length,in_out_dim),bottleneck_dim=bottleneck,out_dim=in_out_dim)

    #Инициализация полносвязной модели
    

    #Обучение и проверка полносвязной модели
    model=train_model(model,train_data_x,train_data_y,output_path)
    results=test_model(model,test_data_x,test_data_y)
    print(f"Результаты для {model_name} MAE - {results['MAE']} MSE - {results['MSE']}")

    #Вычисление feature importance

    fi_base=permutation_feature_importance(model,train_data_x,train_data_y)
    np.save(output_path+f"/{model_name}_feature_importance.npy",fi_base)
    #Рисование графиков
    if in_out_dim>150:
        print("Слишком много пунктов. Нарисовать график fi не удается.")
    else:
        create_feature_importance_chart(fi_base,f"{model_name} feature importance",output_path+f"/{model_name}_fi_chart.png")

    create_feature_importance_map(fi_base,main_dataset.coords,main_dataset.data,f"{model_name} feature importance",output_path+f"/{model_name}_fi_map.png")

    fi_monthes=monthes_permutation_feature_importance(model,main_dataset.data,datetime.date(2025, 1, 1),datetime.date(2025, 12, 31),time_length)
    np.save(output_path+f"/{model_name}_feature_importance_monthes.npy",fi_monthes)

    draw_month_maps(fi_monthes,main_dataset.coords,main_dataset.data,f"{model_name} feature importances по месяцам",output_path+f"/{model_name}_feature_importance_monthes.png")
