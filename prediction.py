from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

import tensorflow as tf
import os
import numpy as np
import cv2
import sqlite3
import time

# select face data class
# ex: face_1
face_class = '4'

gpu_no = "0"  # GPU_index
# setting use GPU run
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
#use_GPU = ['/gpu:0']
#use_GPU = ['/gpu:0','/gpu:1']
N_GPU = '/gpu:0'
I_CPU = '/cpu:0'

# check current project location path
currentPath = os.getcwd()
print("Current Path: " + currentPath)

# select use face class
use_faceclass = 'face_'+face_class

# h5 model storage path
h5_model_modelname = use_faceclass + '_FaceID_model.h5'
h5_model_weightname = use_faceclass + '_FaceID_weight.h5'
h5_model_path = os.path.join(currentPath, 'model', use_faceclass)
get_model = Model()

# predict of face img storage path
predict_faceimg_path = 'predictFaceimg\\'+use_faceclass+'\\'

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

# log output file
outputfile_path = currentPath + '\\log\\'
write_fileName = use_faceclass+"_prediction_output.txt"
full_outputfile_path = os.path.abspath(
    os.path.join(outputfile_path+write_fileName))


# time format settings
time_format = '%Y-%m-%d %H:%M:%S'

# decode one_hot_code
# def one_hot_decode(predict_result):

#    result = np.argmax(predict_result[0])

#    return result

correct_count = 0
err_count = 0
detail_log = ""

# ------------------------------ main mathod ---------------------------
if __name__ == '__main__':

    start_time = time.time()

    print("\n\nWait for configure operation sources...... \n\n")

   # --------------------------- load model ---------------------------
    with tf.device(I_CPU):

        get_model = load_model(h5_model_path+'\\'+h5_model_modelname)

        print('\n\nStart Load Model.')

        print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # check model construction
        print("Model layer length: {}".format(len(get_model.layers)))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # get_model.summary()

    # -------------------------------------------------------------------

    # --------------------------- load face image -----------------------

    with tf.device(N_GPU):

        print('\n\nStart Load Face Image. \n\n')

        temp_log = '''
        --------------------------------------------------------------------------

        This is run keras model predict result log.
        The face dataset use [ {} ] and model use [ {} ].     

        --------------------------------------------------------------------------
        \n\n
        '''.format(use_faceclass, h5_model_modelname)

        print("{}".format(temp_log))

        for dir_list in os.listdir(predict_faceimg_path):

            load_start_time = time.time()

            full_path = os.path.abspath(
                os.path.join(predict_faceimg_path+dir_list))

            if os.path.isdir(full_path):

                for dir_item in os.listdir(full_path):

                    print('Load Face Image:  {} '.format(dir_item))

                    item_path = os.path.abspath(
                        os.path.join(full_path+'\\'+dir_item))

                    #img = image.load_img(full_path)
                    img = cv2.imread(item_path)
                    img = cv2.resize(img, (img_resize, img_resize))
                    #print('Image size: {}'.format(img.size))

                    # image convert to array
                    img_array = image.img_to_array(img)
                    #print('Image shape: {}'.format(img_array.shape))

                    # image shape change dimensions ex: (640,480,3) --> (None,640,480,3)
                    img_array = np.expand_dims(img_array, axis=0)
                    print('Image change dims shape: {}\n'.format(img_array.shape))

                    img_array = img_array.astype('float32')
                    img_array /= 255

                    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    predict_result = get_model.predict(
                        img_array, batch_size=batch_size)
                    faceindex = np.argmax(predict_result[0])

                    SQL_select_syntax = '''
                    SELECT {} FROM {} WHERE {} = '{}'
                    '''.format(column_facename, db_TableName, column_faceindex, faceindex)

                    SQL_run = curs.execute(SQL_select_syntax)
                    SQL_result = curs.fetchall()

                    if len(SQL_result) > 0:

                        filewrite = open(full_outputfile_path, 'w')

                        #print('Face Image Predict result of : {}'.format(SQL_result))

                        load_end_time = '{:.2f}'.format(
                            (time.time() - load_start_time))
                        #print("Speed Time: {}s".format(load_end_time))

                        temp_log += "\n"
                        temp_log += " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
                        temp_log += " Load Face Image:  {}\n".format(dir_item)
                        temp_log += " Face Image Predict result of : {} ,[{}]\n".format(
                            SQL_result[0][0], faceindex)
                        temp_log += " Speed Time: {}s\n".format(load_end_time)
                        temp_log += " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"

                        # temp_log = temp_log + '''\n
                        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n
                        # \n
                        # Load Face Image:  {}\n
                        # Face Image Predict result of : {} ,[{}]\n
                        # Speed Time: {}s\n
                        # \n
                        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        # \n\n\n
                        # '''.format(dir_item,SQL_result,faceindex,load_end_time)

                        # print(type(SQL_result))
                        # print(SQL_result[0][0])
                        # print(dir_list)

                        if dir_list != SQL_result[0][0]:
                            err_count += 1
                            detail_log += ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
                            detail_log += ''' "{}" Predict ----> {} (X)\n'''.format(
                                dir_item, SQL_result[0][0])
                            detail_log += ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n"
                        else:
                            correct_count += 1
                            detail_log += ''' "{}"  Predict ----> {} (O)\n\n'''.format(
                                dir_item, SQL_result[0][0])

                        # filewrite.write('''
                        #   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        #   Load Face Image:  {} \n
                        #   Face Image Predict result of : {} ,[{}]\n
                        #   Speed Time: {}s\n
                        #   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                        #   \n\n\n
                        # '''.format(dir_item,SQL_result,faceindex,load_end_time))
                        # filewrite.close()

                    else:

                        print('Run SQL Error')

                    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

        end_time = '{:.2f}'.format((time.time() - start_time))

        temp_log = temp_log + \
            '''\n\nFull Speed Time: {}s\n\n'''.format(end_time)

        detail_log += '''
        -------------------------------
        Predict Correct count:  {}
        Predict Error   count:  {} 
        -------------------------------
        '''.format(correct_count, err_count)

        print('\n'+detail_log)

        filewrite.write(temp_log + detail_log)
        filewrite.close()

        print("Full Speed Time: {}s".format(end_time))

    # -------------------------------------------------------------------
