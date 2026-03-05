import sys
import os

def choose_files():
    # Проверяем количество аргументов
    if len(sys.argv) != 3:
        print("Ошибка: Необходимо указать два пути к файлам")
        print(f"Использование: {sys.argv[0]} <путь_к_датасету.csv> <путь_к_координатам.csv>")
        print("Пример: python script.py data/dataset.csv data/coordinates.csv")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    coordinates_path = sys.argv[2]
    
    # Проверяем существование файлов
    if not os.path.exists(dataset_path):
        print(f"Ошибка: Файл датасета не найден: {dataset_path}")
        sys.exit(1)
    
    if not os.path.exists(coordinates_path):
        print(f"Ошибка: Файл координат не найден: {coordinates_path}")
        sys.exit(1)
    
    print(f"Файл датасета: {dataset_path}")
    print(f"Файл координат: {coordinates_path}")
    
    return dataset_path,coordinates_path