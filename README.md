# 使用Keras建置CNN神經網絡模型搭配之臉部辨識實例

本專案只是基礎實戰應用筆記，針對模型參數設定配置並沒有特別深入追究 。
運用Keras操作Tensorflow。

本程式專案環境需求：
----------------------------------------------------------------------------------------------------------------------------------
1. 使用Nvidia-GPU
2. Visual Studio 2015 Community
3. Nvidia CUDA 9.0
4. Nvidia cudnn-9.0
5. Python 3.6
6. Tensorflow-GPU
7. Anaconda
8. Visual Studio Code → 可以不安裝，只用來編輯修改程式
9. DroidCam → 將手機鏡頭當作WebCam串流之APP



本程式專案使用Python Library Package：
----------------------------------------------------------------------------------------------------------------------------------
1. tensorflow-gpu==1.11
2. keras
3. sklearn
4. matplotlib
5. numpy
6. opencv-python


本程式專案結構：
----------------------------------------------------------------------------------------------------------------------------------
1. FaceID.py --> 整體訓練流程控制
2. loadImg.py --> 讀取圖片轉換成數據
3. dataset_process.py --> 將資料轉成訓練、驗證、測試資料集
4. model.py --> 建立 CNN 模型框架
5. history_plot.py --> 將Loss、Accuracy訓練過程輸出成一張圖
6. model_logouput.py --> 模型參數輸出
7. prediction_WebCam.py --> 執行臉部圖片預測程序，圖片來源使用手機相機作為WebCam
8. prediction.py --> 執行臉部圖片預測程序
9. face_capture.py --> 運用WebCam進行臉部擷取，產生臉部圖像資料集




