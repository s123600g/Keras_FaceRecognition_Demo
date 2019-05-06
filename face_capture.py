# -*- coding: utf-8 -*-

import os
import cv2 as opcv
import dlib

# -------------------------------global argumentsettings-------------------------------------------
# select face data class
# ex: face_1
face_class = '4'

# check current project location path
currentPath = os.getcwd()
print("Current Path: " + currentPath)

# face image dataSet storage detail root directory
FaceImgDataSet_Path = os.path.join(
    currentPath, 'FaceImg', str('face_'+face_class))
print("FaceImg DataSet Path: " + FaceImgDataSet_Path)

# 擷取臉部圖片存取類型
faceimg_type = ".jpg"

# max number of capture face image
max_capture_faceimg_num = 10

# 設定偵數擷取點,用來存取臉部影像需要用到參數
capturefps = 3

# 圖像擷取索引數
capture_index = 0

# webcam 設定參數
'''
 webcam URL example (droidcam):
 http://192.168.31.119:4747/videostream.cgi?.mjpg
'''
camera = 'http://192.168.31.119:4747/videostream.cgi?.mjpg'
video_width = 960  # 影像寬度
video_height = 720  # 影像高度

# opencv rectangle color 參數
face_recognition_opr_r = 200
face_recognition_opr_g = 35
face_recognition_opr_b = 25

# Dlib face detection instances
dlib_detector = dlib.get_frontal_face_detector()


# ------------------------------ main mathod ---------------------------
if __name__ == '__main__':

    print()

    face_name = input("請輸入臉部標籤：")

    print("輸入臉部標籤：{}".format(face_name))

    # 判斷當前圖片存放根目錄是否存在，如果不存在會返回False在經過not反轉為True
    if not os.path.exists(FaceImgDataSet_Path):

        os.mkdir(FaceImgDataSet_Path)

        print("圖片存放根目錄不存在，需要自動建立.")

    else:

        print("圖片存放根目錄已存在，不需在自動建立.")

    print()

    print("臉部圖像存放位置：{}".format(os.path.join(FaceImgDataSet_Path, face_name)))

    # 判斷該臉部標籤是否存在指定儲存位置底下，如果不存在會返回False在經過not反轉為True
    if not os.path.exists(os.path.join(FaceImgDataSet_Path, face_name)):

        print("['{}']臉部標籤不存在當前指定儲存位置底下".format(face_name))
        os.mkdir(os.path.join(FaceImgDataSet_Path, face_name))

    else:

        print("該臉部標籤目錄已存在指定儲存位置底下.")

    print()

    video_capture = opcv.VideoCapture(camera)  # 連結到指定的webcam並作開啟動作
    video_capture.set(3, video_width)  # 設定影像寬度
    video_capture.set(4, video_height)  # 設定影像高度

    fps = capturefps

    is_first_none = False

    # 開始循環擷取臉部程序
    while True:

        # 判斷圖像擷取索引數是否小於等於設置最大擷取數量，如果條件不成立代表要停止擷取臉部數量
        if capture_index <= max_capture_faceimg_num:

            # 擷取臉部影像
            ret, frame = video_capture.read()

            # print("capture_index： {}".format(capture_index))

            # 偵測人臉
            face_detection = dlib_detector(frame, 0)

            # 判斷偵數是否有達到指定的間格
            if fps == capturefps:

                # 重新初始化fps
                fps = fps - capturefps

                # 判斷是否有偵測到人臉
                if len(face_detection) != 0:

                    # 將圖像擷取索引數+1
                    capture_index += 1

                    # 存取圖像
                    opcv.imwrite(os.path.join(FaceImgDataSet_Path, face_name, str(
                        str(capture_index) + faceimg_type)), frame)

                    print("當前擷取第{}張臉部圖像.".format(capture_index))

                    # 畫出臉部方框
                    opcv.rectangle(
                        frame,
                        # image left & top
                        (
                            face_detection[0].left(),
                            face_detection[0].top()
                        ),
                        # image right & bottom
                        (
                            face_detection[0].right(),
                            face_detection[0].bottom()
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

                    is_first_none = True

            # 判斷是否為初次改為當前未偵測到臉部狀態
            if is_first_none:

                print("當前未偵測到臉部.......")

                is_first_none = False

            # 將fps +1
            fps += 1

        else:

            print("結束擷取臉部圖像.")

            # When everything is done, release the capture
            opcv.destroyAllWindows()

            break

        # 判斷是否有按下'q'鍵，代表要終止程序
        if opcv.waitKey(1) & 0xFF == ord('q'):

            video_capture.release()

            print("\n終止臉部識別程序....")

            break

        # 顯示視窗畫面
        opcv.imshow('WebCamera Frame', frame)
