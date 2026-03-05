import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_feature_importance_chart(feature_importance:np.array,title:str,save_fig_path:str=None):
    plt.figure(figsize=(4.01, 4.01))
    fi_plot_data=(feature_importance-feature_importance.min())/feature_importance.max()/2
    for i in range(feature_importance.shape[0]):
        for j in range(feature_importance.shape[1]):
            plt.scatter(i, j, marker='s',s=330, color=(1-fi_plot_data[i,j],1-fi_plot_data[i,j],1-fi_plot_data[i,j]))
    plt.title(title)
    plt.xlabel("Перемешиваемые столбцы")
    plt.ylabel("Считываемые столбцы")
    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    plt.show()

def create_feature_importance_map(feature_importance:np.array,coords:pd.DataFrame,dataset:pd.DataFrame,title:str,save_fig_path:str=None):
    forfi=pd.DataFrame({"site":dataset.drop("datetime",axis=1).columns,"fi":feature_importance.mean(axis=1)})
    result_data=coords.copy()
    result_data["site"]=result_data["site"].astype(str)
    df=pd.merge(result_data,forfi,"inner","site")
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Добавляем базовые географические элементы
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

    # Добавляем сетку координат
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False   # убираем подписи сверху
    gl.right_labels = False # убираем подписи справа

    # Создаем цвета в градации серого на основе fi
    # Нормализуем значения fi от 0 до 1 для цветовой шкалы
    norm_fi = (df['fi'] - df['fi'].min()) / (df['fi'].max() - df['fi'].min())
    # Инвертируем, чтобы большие значения были темнее (или не инвертируйте, как вам нужно)
    colors = plt.cm.Greys(1 - norm_fi)  # 1 - norm_fi для темных = хорошие

    # Рисуем точки
    scatter = ax.scatter(df['lon'], df['lat'], 
                        c=colors,           # цвета в градации серого
                        s=100,               # размер точек
                        edgecolor='black',   # обводка точек
                        linewidth=0.5,       # толщина обводки
                        transform=ccrs.PlateCarree(),
                        zorder=5)            # чтобы точки были поверх карты

    # Добавляем подписи к точкам
    #for idx, row in df.iterrows():
    #    ax.text(row['lon'] + 0.5, row['lat'] + 0.5, row['site'],
    #            transform=ccrs.PlateCarree(),
    #            fontsize=9,
    #            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Добавляем цветовую шкалу
    # Создаем объект для colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, 
                            norm=plt.Normalize(vmin=df['fi'].min(), 
                                                vmax=df['fi'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Значение fi (чем выше, тем лучше)', fontsize=11)

    # Настраиваем заголовок
    plt.title(title, fontsize=14, pad=20)

    # Автоматически определяем границы карты по данным
    ax.set_extent([df['lon'].min() - 5, df['lon'].max() + 5,
                df['lat'].min() - 3, df['lat'].max() + 3])

    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.show()


def draw_month_maps(feature_importance_months:np.array,coords:pd.DataFrame,dataset:pd.DataFrame,title:str,save_fig_path:str):
    fi_values=feature_importance_months.mean(axis=-1)

    # Названия для каждой карты (12 строк)

    df_sites=coords.copy()
    # Названия для каждой карты (12 строк)
    map_titles = [
        "Январь",
        "Февраль",
        "Март",
        "Апрель",
        "Май",
        "Июнь",
        "Июль",
        "Август",
        "Сентябрь",
        "Октябрь",
        "Ноябрь",
        "Декабрь"
    ]

    # СОЗДАЕМ ФИГУРУ
    fig = plt.figure(figsize=(22, 26))

    # Создаем сетку субплотов
    for i in range(12):
        # Создаем субплот
        ax = fig.add_subplot(4, 3, i+1, projection=ccrs.PlateCarree())
        
        # Добавляем элементы карты
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        
        # Сетка
        ax.gridlines(draw_labels=False, linestyle=':', alpha=0.3)
        
        # Данные
        current_fi = fi_values[i]
        
        # Нормализация
        if current_fi.max() > current_fi.min():
            norm_fi = (current_fi - current_fi.min()) / (current_fi.max() - current_fi.min())
        else:
            norm_fi = np.zeros(len(current_fi))
        
        # Цвета
        colors = plt.cm.Greys(norm_fi)
        
        # Точки
        scatter = ax.scatter(df_sites['lon'], df_sites['lat'], 
                            c=colors, s=40, edgecolor='black',
                            linewidth=0.3, transform=ccrs.PlateCarree(),
                            zorder=5, alpha=0.9)
        
        # Заголовок
        ax.set_title(map_titles[i], fontsize=10, pad=8)
        
        # ВАЖНО: Используем make_axes_locatable для создания colorbar справа от графика
        # Это гарантирует, что все графики будут одинакового размера
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        
        # Создаем colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, 
                                norm=plt.Normalize(vmin=current_fi.min(), 
                                                    vmax=current_fi.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=8)

    # Устанавливаем границы для всех карт
    margin_lon = 0.5
    margin_lat = 0.5
    for ax in fig.axes:
        # Проверяем, что это географическая ось (имеет projection)
        if hasattr(ax, 'projection'):
            ax.set_extent([df_sites['lon'].min() - margin_lon, 
                        df_sites['lon'].max() + margin_lon,
                        df_sites['lat'].min() - margin_lat, 
                        df_sites['lat'].max() + margin_lat])

    # Общий заголовок
    fig.suptitle('Сравнение 12 показателей на карте\n(индивидуальные шкалы)', 
                fontsize=16, y=0.98)

    # Настраиваем отступы
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.95,      # отступ сверху
        bottom=0.05,    # отступ снизу
        left=0.05,      # отступ слева
        right=0.95,     # отступ справа
        hspace=0.01,     # ★ горизонтальное расстояние между рядами (очень мало)
        wspace=0.1      # ★ вертикальное расстояние между колонками (очень мало)
    )
    plt.savefig(save_fig_path)
    plt.show()