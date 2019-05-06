# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

import tensorflow as tf
import os
import numpy as np
import cv2 as opcv
import dlib
import sqlite3
import time

# select face data class
# ex: face_1
face_class = '4'

# 自動抓取當前程式所在位置
currentPath = os.getcwd()
print("Current Path: " + currentPath)

# webcam 設定參數
'''
 webcam URL example (droidcam):
 http://192.168.31.119:4747/videostream.cgi?.mjpg
'''
camera = 'http://192.168.31.119:4747/videostream.cgi?.mjpg'
video_width = 960  # 影像寬度
video_height = 720  # 影像高度

# opencv rectangle color 參數
face_recognition_opr_r = 255
face_recognition_opr_g = 75
face_recognition_opr_b = 0

# Dlib face detection instances
dlib_detector = dlib.get_frontal_face_detector()

# select use face class
use_faceclass = 'face_'+face_class

# h5 model storage path
h5_model_modelname = use_faceclass + '_FaceID_model.h5'
h5_model_weightname = use_faceclass + '_FaceID_weight.h5'
h5_model_path = os.path.join(currentPath, 'model', use_faceclass)
get_model = Model()

batch_size = 32
img_resize = 96

# SQL arguments settings
SQLite_path = os.path.join(currentPath, 'facelabeldb', str("face_"+face_class))
SQLite_name = "facelabel.db3"
dbconn = sqlite3.connect(os.path.join(SQLite_path, SQLite_name))
curs = dbconn.cursor()
db_TableName = 'facelabel'
column_faceindex = 'faceindex'
column_facename = 'facename'

# time format settings
time_format = '%Y-%m-%d %H:%M:%S'


def recognition_process(frame, get_model):

    recognition_start_time = time.time()

    # 顯示原始圖像矩陣維度
    # print('Image shape: {}'.format(frame.shape))

    # 改變圖像矩陣維度，增加在第一個位置 ex: (640,480,3) --> (1,640,480,3)
    img_array = np.expand_dims(frame, axis=0)
    # print('Image change dims shape: {}\n'.format(img_array.shape))

    img_array = img_array.astype('float32')
    img_array /= 255

    # 執行模型預測
    predict_result = get_model.predict(img_array, batch_size=batch_size)

    # 取出模型預測結果索引，條件為取出最高值
    faceindex = np.argmax(predict_result[0])

    # for read in predict_result:
    #     print("{}".format(read))

    # print("faceindex： {}".format(faceindex))
    # print("Confidence: {:.2f}".format(predict_result[0][faceindex]))

    # 設置查詢SQLite 資料庫語法
    SQL_select_syntax = '''
                    SELECT {} FROM {} WHERE {} = '{}'
                    '''.format(column_facename, db_TableName, column_faceindex, faceindex)

    # 執行SQL查詢語法
    SQL_run = curs.execute(SQL_select_syntax)

    # 取得SQL查詢語法結果，並且將其轉型成list型態
    SQL_result = curs.fetchall()

    recognition_result = ""
    recognition_end_time = 0
    confidences = 0

    if len(SQL_result) > 0:

        #print('Face Image Predict result of : {}'.format(SQL_result))

        recognition_end_time = '{:.2f}'.format(
            (time.time() - recognition_start_time))
        #print("Speed Time: {}s".format(load_end_time))

        recognition_result = SQL_result[0][0]  # 取得識別標籤結果
        confidences = "{:.2f}".format(predict_result[0][faceindex])  # 可信度取得

        print("臉部預測識別結果： ['{}'] ,  可信度：{} , 辨識花費時間： {}s".format(
            recognition_result, confidences, recognition_end_time))

    else:

        print('Run SQL Error')

        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

    return recognition_result, confidences, recognition_end_time


# ------------------------------ main mathod ---------------------------
if __name__ == '__main__':

    print("\n\nWait for configure operation sources...... \n\n")

    # --------------------------- load model ----------------------------
    get_model = load_model(h5_model_path+'\\'+h5_model_modelname)  # 載入模型

    print('\n\nStart Load Model.')

    # print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # check model construction
    # print("Model layer length: {}".format(len(get_model.layers)))
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    get_model.summary()
    # -------------------------------------------------------------------

    print('\n\nStart Load Face Image. \n\n')

    # --------------------------- load face image -----------------------

    video_capture = opcv.VideoCapture(camera)  # 連結到指定的webcam並作開啟動作
    video_capture.set(3, video_width)  # 設定影像寬度
    video_capture.set(4, video_height)  # 設定影像高度

    if video_capture.isOpened == False:  # 判斷webcam是否真的有連結到並作開啟動作

        print('Can not connect camera.')

    else:

        is_first_none = False

        while True:

            start_time = time.time()

            # 擷取臉部影像
            ret, frame = video_capture.read()

            # 偵測人臉
            face_detection = dlib_detector(frame, 0)

            # 判斷是否有偵測到人臉
            if len(face_detection) != 0:

                # print("face_detection：")

                # 讀取偵測到臉部資訊
                for read_item in face_detection:

                    # print("Top：{}".format(read_item.top()), end=' , ')
                    # print("Left：{}".format(read_item.left()), end=' , ')
                    # print("Bottom：{}".format(read_item.bottom()), end=' , ')
                    # print("Right：{}".format(read_item.right()))

                    print()

                    # 將圖像大小重新調整
                    img = opcv.resize(frame, (img_resize, img_resize))

                    # 取得辨識預測結果
                    recognition_result, confidences, recognition_end_time = recognition_process(
                        img, get_model)

                    # 畫出臉部方框
                    opcv.rectangle(
                        frame,
                        # image left & top
                        (
                            read_item.left(),
                            read_item.top()
                        ),
                        # image right & bottom
                        (
                            read_item.right(),
                            read_item.bottom()
                        ),
                        # rectangle color arguments(b,g,r)
                        (
                            face_recognition_opr_b,
                            face_recognition_opr_g,
                            face_recognition_opr_r
                        ),
                        # shift 方框粗度
                        2
                    )

                    # 輸出影像文字資訊再臉部方框底下
                    opcv.putText(
                        frame,
                        "Face Name:[ {} ] Confidences：{}  | Speed:[ {}s ]".format(
                            recognition_result, confidences, recognition_end_time),
                        (read_item.left(), read_item.bottom() + 20),
                        opcv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                is_first_none = True

            else:

                # 判斷是否為初次改為當前未偵測到臉部狀態
                if is_first_none:

                    print("當前未偵測到臉部.......")

                    is_first_none = False

            # 顯示視窗畫面
            opcv.imshow('WebCamera Frame', frame)

            # 判斷是否有按下'q'鍵，代表要終止程序
            if opcv.waitKey(1) & 0xFF == ord('q'):

                video_capture.release()

                print("\n終止臉部識別程序....")

                break

        # When everything is done, release the capture
        opcv.destroyAllWindows()

    # -------------------------------------------------------------------
