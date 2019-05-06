# -*- coding: utf-8 -*-

from FaceID import face_class

import os
import cv2
import numpy as np
import sqlite3
import shutil


# get current project location path
currentPath = os.getcwd()

# SQL arguments settings
SQLite_path = os.path.join(currentPath, 'facelabeldb', str("face_"+face_class))
SQLite_name = "facelabel.db3"
SQLite_source = os.path.join(
    currentPath, 'facelabeldb', 'db_sources', SQLite_name)

dbconn = sqlite3.connect(os.path.join(SQLite_path, SQLite_name))
curs = dbconn.cursor()
db_TableName = 'facelabel'
column_faceindex = 'faceindex'
column_facename = 'facename'


def load_faceimg(faceimgpath, imgtype, img_resize):

    load_index = 0
    images = []  # face image
    labels = []  # face label
    label_class = []

    print("Start Scanning faceimg folder image class.\n")

    # run SQL DB Table clearn
    SQL_delete_syntax = '''
    DELETE FROM {}
    '''.format(db_TableName)
    SQL_run = curs.execute(SQL_delete_syntax)

    if SQL_run:

        dbconn.commit()

    else:

        print('Run SQL_delete_syntax Faild.')

    # scan faceimg folder
    for dir_item in os.listdir(faceimgpath):

        #print("No."+str(load_index)+" [ "+dir_item+" ]")
        print("\nNo.{} [ {} ]".format(str(load_index), dir_item))

        load_index += 1
        label_class.append(dir_item)

        # join root directory and class folder name
        full_path = os.path.abspath(os.path.join(faceimgpath) + dir_item)
        print("Full Detail Path = {}".format(full_path))

        class_name = dir_item
        num_img = 0

        # read image
        for dir_item in os.listdir(full_path):

            # print(dir_item)
            num_img += 1
            img_path = ""

            if dir_item.endswith(imgtype)and os.path.exists(str(full_path)):

                img_path = os.path.abspath(
                    os.path.join(full_path) + "\\" + dir_item)
                # print(img_path)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (img_resize, img_resize))

                # get image size
                img_width, img_height, channels = image.shape

                # if num_img < 10: # 1~9

                #    dir_item = str(dir_item).rstrip("_000" + str(num_img) + ".jpg").rstrip()

                # elif num_img >= 10 and num_img < 100: # 10~99

                #    dir_item = str(dir_item).rstrip("_00" + str(num_img) + ".jpg").rstrip()

                # elif num_img >= 100 and num_img < 1000: # 100~999

                #    dir_item = str(dir_item).rstrip("_0" + str(num_img) + ".jpg").rstrip()

                # else: # 10~99

                #    dir_item = str(dir_item).rstrip("_" + str(num_img) + ".jpg").rstrip()

                # add image to array
                images.append(image)
                labels.append(class_name)

        print("Image Quantity: {}".format(num_img))

    return images, labels, img_width, img_height, channels, load_index, label_class


def char_to_code(labels, label_class, n_class):

    # for index,item in label_class:

    #    # run SQL_Insert_syntax
    #    SQL_Insert_syntax = '''
    #    INSERT INTO {}('{}','{}')
    #    VALUES('{}','{}')
    #    '''.format(db_TableName,column_faceindex,column_facename,str(index),str(item))
    #    SQL_run = curs.execute(SQL_Insert_syntax)

    #    if SQL_run:
    #        dbconn.commit()
    #    else:
    #        print ('Run SQL_Insert_syntax Faild.')

    for index, classname in label_class:

        labels = np.array(
            [int(index) if label == classname else label for label in labels])

        # run SQL_Insert_syntax
        SQL_Insert_syntax = '''
        INSERT INTO {}('{}','{}')
        VALUES('{}','{}')
        '''.format(db_TableName, column_faceindex, column_facename, str(index), str(classname))
        SQL_run = curs.execute(SQL_Insert_syntax)

        if SQL_run:
            dbconn.commit()
        else:
            print('Run SQL_Insert_syntax Faild.')

    return labels
