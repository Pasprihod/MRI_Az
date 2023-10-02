#УДАЛЕНИЕ ШУМА С МРТ

import os
import cv2
from keras.models import load_model
import numpy as np
import pydicom as dicom # библиотека для работы с dicom
import datetime
import time

h = 256 # требуемая размерность изображений
w = 256
MRI = 'S:/AZ-360-2'
files_checked = [] # список имен проверенных файлов
day = datetime.datetime.now().day
month_idx = datetime.datetime.now().month
year = datetime.datetime.now().year
months = ('Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь','Декабрь')
if day < 10:
  FOLDER = f'{MRI}/{year}/{months[month_idx-1]}/0{day}'
else:
  FOLDER = f'{MRI}/{year}/{months[month_idx-1]}/{day}'
FOLDER_model = 'C:/AZIMUT/AZIMUT_AI/unet.h5'
unet = load_model(FOLDER_model)

# нормализация к 0..1

def normal(image): 
  max_pixel = np.max(image) 
  min_pixel = np.min(image)
  if max_pixel > min_pixel:
    image_norm = (image - min_pixel) / (max_pixel - min_pixel) 
    return image_norm, max_pixel, min_pixel

# Предикт, оценка полученного уровня шума (среднее, стан. отклонение)
def predict(model, files_for_recon, FOLDER,w=w, h=h):
  
  dicom_list = [] # список dicom файлов с зашумленными изображениями из папки path
  image_list = [] # список зашумленных изображений из dicom файлов из папки path
  max_min_list = [] # список max и min значения пикселей зашумленных изображений из dicom файлов из папки path 
                    # для возврата очищенных изображений  к первоначальной шкале
  w_h_list =[] # список с размерностями входных изображений (для их последующего восстановления)
 
  for f in sorted(files_for_recon):
    d = dicom.dcmread(os.path.join(FOLDER,f)) 
    # изменение полей в DICOM файле, чтобы при просмотре очищенные изображения отличались
    # от первоначальных зашумленных
    v1_number = int(d[0x0008,0x0018].value.split('.')[-1])
    v2_number = int(d[0x0008,0x0018].value.split('.')[-2])+1
    v3_number = int(d[0x0008,0x0018].value.split('.')[-3])
    d[0x0008,0x0018].value = '.'.join(d[0x0008,0x0018].value.split('.')[:-3]) +'.' + str(v3_number)+'.'+ str(v2_number)+'.'+ str(v1_number)
    d.file_meta[0x0002,0x0003].value = d[0x0008,0x0018].value
    d[0x0020,0x000e].value = '.'.join(d[0x0008,0x0018].value.split('.')[:-3]) +'.' + str(v3_number)+'.'+ str(v2_number) 
    #d[0x0008,0x103e].value = d[0x0008,0x103e].value + '_AI_denoising' # название протокола
    d[0x0020,0x4000].value = 'AI_recon'
    dicom_list.append([d,f])
    # извлечение ndarray изображения из dicom
    image = d.pixel_array
    w_real = image.shape[0]
    h_real = image.shape[1]
    w_h_list.append((w_real, h_real))
    if w_real != w or h_real != h: 
       image = cv2.resize(image,(w, h), interpolation=cv2.INTER_AREA)
    # приведение к шкале 0,1 для подачи на вход модели, сохранение max, min пикселей для 
    # последующего восстановления шкалы  
    image_norm, max, min = normal(image)
    # создание списков  
    image_list.append(image_norm.reshape(image.shape[0], image.shape[1],1))
    max_min_list.append((max, min))
  max_min_array = np.array(max_min_list, dtype='float')
  image_array = np.array(image_list)
  pred_image = model.predict(image_array)
  for i in range(image_array.shape[0]):
    max,min = max_min_array[i]
    # восстановление шкалы очищенного изображения
    image = pred_image[i]
    image = image.reshape(image.shape[0], image.shape[1])*(max-min) + min
    # перезапись изображения на восстановленное
    if w_h_list[i][0] != w or w_h_list[i][1] != h:
       image = cv2.resize(image,(w_h_list[i][1], w_h_list[i][0]), interpolation=cv2.INTER_LINEAR)
    dicom_list[i][0].PixelData = image.astype(np.uint16).tobytes()
  return dicom_list # [dicom, имя файла]
  
# Запись в dicom формат для дальнейшего использования в специальном dicom-просмотровщике
def write_result_in_dicom(dicom_list, path_for_write):
  for i in range(len(dicom_list)):
    dicom_name = dicom_list[i][1]
    dicom_list[i][0].save_as(os.path.join(path_for_write,f"{dicom_name[:-4]}_AI.DCM"))

def recon(files_checked,FOLDER, model):
  t1 = time.time()
  files_for_recon = [] # список имен файлов для predict
  for f in sorted(os.listdir(FOLDER)):
    file_path = os.path.join(FOLDER, f)
    # Далее ПРОВЕРКА:
    # 1. это файл (не папка CDS), 2. этот файл еще не обрабатывался,
    # 3. он не в списке проверенных, 4. он сам не реконструкция, 5. толщина среза < 6 мм
    if os.path.isfile(file_path) and f not in files_checked and f[-6:-4] != 'AI' and not os.path.exists(f"{file_path[:-4]}_AI.DCM") :
      d = dicom.dcmread(file_path)
      if d[0x0008, 0x103e].value in ('T2W TRV') and d[0x0008, 0x1030].value in ('Головной мозг') and d[0x0028, 0x0010].value <= h and d[0x0028, 0x0011].value <= w and float(d[0x0018, 0x0050].value) < 6:
        files_for_recon.append(f)
        files_checked.append(f)
  print('Время просмотра файлов и подготовки списков: ', round(time.time()-t1,2))

  if len(files_for_recon) > 0:
    print('Реконструкция запущена...Количество файлов для реконструкции: ', len(files_for_recon))
    dicom_list = predict(model, files_for_recon, FOLDER)
    write_result_in_dicom(dicom_list, FOLDER)
    print(f'Реконструкция завершена')
  else:
    print('Новых файлов нет')
  print('Общее время: ', round(time.time()-t1,2), ' с')
  files_checked = list(set(files_checked))

files_count = 0# количество файлов в папке
files_checked = [] # список имен проверенных подходящих файлов

while True:
  if os.path.exists(FOLDER):
    if len(os.listdir(FOLDER)) > files_count:
      recon(files_checked, FOLDER=FOLDER, model = unet)
      files_count = len(os.listdir(FOLDER)) 
    elif len(os.listdir(FOLDER)) < files_count:
      files_count = len(os.listdir(FOLDER))
  time.sleep(5)