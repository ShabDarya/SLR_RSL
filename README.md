
# SLR_RSL


## Abstract

Данный проект реализует систему распознавания русского жестового языка (РЖЯ) в режиме реального времени с использованием датасета [SLOVO](https://github.com/hukenovs/slovo/tree/main?tab=readme-ov-file).

[Полный текст ВКР](https://drive.google.com/file/d/1Jmi00wSI_S_P7Lo1LKy8v58mASWTUD1i/view?usp=sharing)





## Структура проекта


```bash
SLR_RSL/
├── Application/                                   # Папка с файлами для приложения
│   ├── best_model_epoch_266_top1_33.64.pth        # Веса лучшей модели
│   ├── face_landmarker_v2_with_blendshapes.task   # Модель для расп-ния лицаMediaPipe)
│   ├── hand_landmarker.task                       # Модель для распознавания рук(MediaPipe)
│   ├── idx2label.json                             # Файл маппинга классов
│   ├── pose_landmarker.task                       # Модель для расп-ния позы(MediaPipe)
├── Code_app/                                      # Файлы необходимые для 
│   ├── best_model_epoch_266_top1_33.64.pth        # Веса лучшей модели      
│   ├── class_model.py                             # Класс модели
│   ├── code_for_installer.txt                     # Код для компиляции
│   ├── icon.ico                                   # Иконка приложения
│   ├── idx2label.json                             # Файл маппинга классов
│   └── model_file.py                              # Файл для вызова функций модели
├── data/                                          # Папка с примерами данных
│   ├── random_test_sample.npy                     # Пример тестовых данных
│   └── random_train_sample.npy                    # Пример тренировочных данных
├── Analyse_dataset.ipynb                          # Jupyter ноутбук с анализом датасета
├── LICENSE                                        # Файл лицензии
├── README.md                                      # Файл описания проекта   
├── Training_model.ipynb                           # Ноутбук с обучением модели
├── Video Processing.ipynb                         # Ноутбук с извлечением ключевых точек
└── requirements.txt                               # Необходимые библиотеки


```                                              
## Запуск программы

Для запуска программы необходимо скачать папку [Application](https://github.com/ShabDarya/SLR_RSL/tree/main/Application), [RGL.exe](https://drive.google.com/drive/folders/1Fq9W1oypQ5w5qM5FWiNyAdRnJH5YwBpp?usp=drive_link) и запустить приложение. Оно уже готово к использованию.

Для использования ноутбуков нужно:

0. (Опционально) Создать и активировать новую среду используя [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=3.12

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Установить все необходимые библиотеки

   ```bash
   pip install -r requirements.txt
   ```
