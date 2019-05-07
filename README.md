# 使用Keras建置CNN神經網絡模型搭配之臉部辨識實例

本專案只是基礎實戰應用筆記，針對模型參數設定配置並沒有特別深入追究 。
運用Keras操作Tensorflow。

本程式專案環境需求與安裝順序：
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


本程式專案使用Python Package：
----------------------------------------------------------------------------------------------------------------------------------
1. tensorflow-gpu==1.11
2. keras
3. sklearn
4. matplotlib
5. numpy
6. opencv-python

#### 安裝方法1：使用讀取清單文件安裝
> pip install -r requirements.txt <br/>

如果選擇此操作需要再獨自安裝dlib套件， 此為臉部偵測需要用到套件庫。
<br/>
> pip install dlib/dlib-19.8.1-cp36-cp36m-win_amd64.whl

#### 安裝方法2：使用pip指令獨立安裝
1.keras<br/> 
> pip install keras<br/>
2.sklearn<br/>
> pip install sklearn<br/>
3.matplotlib<br/>
> pip install matplotlib<br/>
4.numpy<br/>
> pip install numpy<br/>
5.opencv-python<br/>
> pip install opencv-python<br/>
6.dlib<br/>
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




