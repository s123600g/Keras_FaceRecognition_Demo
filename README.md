# 使用Keras建置CNN神經網絡模型搭配之臉部辨識實例

本專案只是基礎實戰應用筆記，針對模型參數設定配置並沒有特別深入追究 。<br/>
運用Keras操作Tensorflow。

[License](https://github.com/s123600g/Keras_FaceRecognition_Demo/blob/master/LICENSE)

執行環境需求與安裝順序：
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

#### Visual Studio 2015 Community 官網下載資源：
[Visual Studio 2015 Community](https://visualstudio.microsoft.com/vs/older-downloads/)
Install Visual C++ Build Tools 2015

#### Nvidia CUDA 9.0 官網下載資源：
[CUDA Toolkit 9.0 Downloads](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10)

#### Nvidia cudnn-9.0 官網下載資源：
[cuDNN Download](https://developer.nvidia.com/rdp/form/cudnn-download-survey)

#### 關於Tensorflow-GPU安裝方法，可參考官網資源：
[Build from source on Windows](https://www.tensorflow.org/install/source_windows)

#### Anaconda 官網下載資源：
[Anaconda Download](https://www.anaconda.com/distribution/#download-section)

#### 安裝CUDA與cudnn 9.0 注意事項
1. 針對CUDA安裝前，需要先安裝好 Visual Studio 2015 Community。
2. 安裝CUDA完畢後，需要將cudnn-9.0資料夾內所有檔案目錄複製到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0 底下。
3. 在系統環境變數路徑需要加上 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin


使用Python Package：
----------------------------------------------------------------------------------------------------------------------------------
1. tensorflow-gpu==1.11
2. keras
3. sklearn
4. matplotlib
5. numpy
6. opencv-python
7. dlib

#### 安裝方法1：使用讀取清單文件安裝
> pip install -r requirements.txt <br/>

如果選擇此操作需要再獨自安裝dlib套件， 此為臉部偵測需要用到套件庫。
<br/>
> pip install dlib/dlib-19.8.1-cp36-cp36m-win_amd64.whl

#### 安裝方法2：使用pip指令獨立安裝
1. keras <br/>
> pip install keras

2. sklearn <br/>
> pip install sklearn 

3. matplotlib<br/>
> pip install matplotlib

4. numpy<br/>
> pip install numpy

5. opencv-python<br/>
> pip install opencv-python

6. dlib(此為臉部偵測需要用到套件庫) <br/>
> pip install dlib/dlib-19.8.1-cp36-cp36m-win_amd64.whl<br/>



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

Tensorflow 模型訓練參數設定：
----------------------------------------------------------------------------------------------------------------------------------
1. batch_size ：23
2. epochs ：84
3. ImageDataGenerator
4. Adam ( learn_rate : 0.00015 )
5. steps_per_epoch：18(432/23)	→ steps_per_epoch = (len(train_images) / batch_size)


圖像資料集操作設置：
----------------------------------------------------------------------------------------------------------------------------------
圖像資料集分別如下：
1. 訓練圖像資料集放置目錄：FaceImg/
2. 預測圖像資料集目錄：predictFaceimg/

上面兩者資料夾內部都會有以’face_數字’為命名資料夾，可以用來指定使用不同臉部圖像資料集區塊，臉部圖像要放置在以’face_數字’為命名資料夾內對應的識別名稱資料夾之中。<br/>
在 FaceID.py、prediction.py、prediction_WebCam.py、face_capture.py內，會有一個變數命名為'face_class'，此為操作要使用哪一個臉部圖像資料集區塊。

以 face_class = '4' 為例，此為代表要操作使用內部以'face_4'為命名臉部圖像資料集區塊資料目錄，內部有六位不同識別名稱資料夾，放置各自對應臉部圖像。
圖像資料集分別如下：
> 1. 訓練圖像資料集放置目錄：FaceImg/face_4
> 2. 預測圖像資料集目錄：predictFaceimg/face_4

face_capture.py 會自動識別是有存在'face_數字'為命名，臉部圖像資料集區塊資料目錄，如果不存在就會自動建立。

WebCam設置：
----------------------------------------------------------------------------------------------------------------------------------
以手機鏡頭作為串流WebCam媒體，使用 DroidCam Wireless Webcam 此手APP作為WebCam串流控制。<br/>

[DroidCam Wireless Webcam](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam&hl=zh_TW)

#### 需注意!!手機跟電腦需要在同一個網路內，也就是在相同區域網段內
如果執行prediction_WebCam.py 與 face_capture.py 兩者程式，需要先設置內部camera連結WebCam URL變數，語法格式如下：
> http://手機區域網路IP:4747/videostream.cgi?.mjpg



Anaconda 執行程式操作：
----------------------------------------------------------------------------------------------------------------------------------
#### 開啟 Anaconda Prompt
開啟 Anaconda Prompt (Windows)，一開始打開會是在(base)預設虛擬環境底下，如果要建立一個虛擬環境，需要先離開(base)預設虛擬環境底下。

#### 建立虛擬環境
建立虛擬環境指令：conda create --name [env_name] [python_env] <br/>
假設要建立一個虛擬環境名稱為KerasFaceID，並且python環境為3.6<br/>
> conda create --name KerasFaceID python=3.6

#### 進入虛擬環境
進入虛擬環境指令：activate [env_name] <br/>
假設要進入一個虛擬環境名稱為KerasFaceID <br/>
> activate KerasFaceID

#### 離開虛擬環境
離開虛擬環境指令：conda deactivate  [env_name] <br/>
假設要離開一個虛擬環境名稱為KerasFaceID <br/>
> conda deactivate

#### 執行程式專案在虛擬環境底下
模型訓練程式：FaceID.py <br/>
> python FaceID.py

臉部識別程式：prediction.py 或 prediction_WebCam.py (使用WebCam)
> python prediction.py <br/>
> python prediction_WebCam.py

執行臉部擷取程式：face_capture.py(使用WebCam)
> python face_capture.py
