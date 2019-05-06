# 使用Keras建置CNN神經網絡模型搭配Tensorflow之臉部辨識實例

本專案只是基礎實戰應用筆記，針對模型參數設定配置並沒有特別深入追究 。

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




