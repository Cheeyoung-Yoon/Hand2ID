# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 00:20:57 2021

@author: parth
"""

######Curently, tested to create ML modelin loop, working on how to find the adequate hyper-parameter in fast manner.
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import tensorflow  as tf
import datetime
import math
import random

import matplotlib.pyplot as plt
import csv
from numba import jit, cuda
# In[function of saving DF]

####file creater:
    ##checking for folder, if DNE then create
def check_dir(file_name,records,names):
    directory = os.path.dirname(file_name)
    if not os.path.exists(file_name):
        try:
            os.makedirs(directory)
        except:
            pass
    
        names.to_csv(file_name,index=False,header=False,encoding="utf-8")
        #dat.to_csv(file_name,index=False,header=False,encoding="cp949")
        
####save created data as csv, creating folder and file       
def save(file_name, records,names):
    check_dir(file_name,records,names)
    records.to_csv(file_name, mode='a',index=False,header=False,encoding="utf-8")
    #records.to_csv(file_name, mode='a',index=False,header=False,encoding="cp949-8")

# In[img to cor]
##whole image to corrdinates
##change image file to corrdinates, 
###1. change image to black and white
###2. find the only black pixel as corrdinate data,
def img2cor(im):
    begin_time = datetime.datetime.now()
    width, height = im.size
    pixel_values = list(im.getdata())
    pixel_flat = [num for elem in pixel_values for num in elem]
    av = sum(pixel_flat)/len(pixel_flat)
    
    pixel_values = np.array(im)
    whiteval = av*0.85
    y= 0
    dat2 = pd.DataFrame([["X","Y"]])
    for i in range(0,height):
        for j in range(0,width): 
            pixval = sum(pixel_values[i][j])/2
            if (pixval > whiteval):
                y = y+1
            else:
                tmp = pd.DataFrame([[j,height-i]])
                dat2= dat2.append(tmp)
                del tmp
    dat2=dat2.rename(columns=dat2.iloc[0])
    dat2 = dat2.iloc[1: , :]
    return dat2
    print("img2corr time: " + str(datetime.datetime.now() - begin_time))
    
    


# dat.X = (dat.X-min(dat.X))*100/max(dat.X)
# dat.Y = (dat.Y-min(dat.Y))*100/max(dat.Y)

# In[split corrs]:
##As the first takes the whole data, it splits the data into part detecting the white spaces.
def splitter(line_dat, Y_range,min_avg_x):
    line_number = 0
    all_elements = []
    line_dat.columns=['X','Y']
    line_dat.sort_values("Y")
    
    Y_group = line_dat.groupby(pd.cut(line_dat["X"], np.arange(0,max(line_dat.X),Y_range))).size()
    Y_group = Y_group.to_frame()
    Y_group_asc=Y_group.reset_index()
    Y_group_dec=Y_group.sort_values('X',ascending=False)
    Y_group_dec=Y_group_dec.reset_index()
    
    Xmin = 0
    Xmax = 0

    #Y_group[0].sum()/height
    #     Y_group[0].mean()*mean_adj
    Xrange = pd.DataFrame()
    
    Xmins = []
    Xmaxs = []
    for i in range(0,len(Y_group)):
    
        if Y_group_asc[0][i] <min_avg_x:
            Xmin = 0
    
        if Y_group_asc[0][i] >= min_avg_x and Xmin == 0:
            left = Y_group_asc['X'][i].left
            Xmin = left
            Xmins += [left]
    
        if Y_group_dec[0][i] <min_avg_x:
            Xmax = 0
    
        if Y_group_dec[0][i] >= min_avg_x and Xmax == 0:
            right = Y_group_dec['X'][i].right
            Xmax = right
            Xmaxs += [right]
    
        Xmaxs.sort()
    
    Xrange = pd.DataFrame({'min':Xmins,'max':Xmaxs})
    ###get each line data to elements[]
    elements = []
    for j in range(0,len(Xrange)):
        elements +=[line_dat[(line_dat.X>Xrange['min'][j])& (line_dat.X<Xrange['max'][j])].values.tolist()]
    
    for element_number in range(0,len(elements)):    
        all_elements+= [[[line_number,element_number],[elements[element_number]]]]
    
    return all_elements

# In[spinning]
###hand writing never written in same angle, so it spins slitly to add more training data
def spin(degree,df):
    spin_df = pd.DataFrame([["X","Y"]])
    rad = np.deg2rad(degree)
    
    origin_x = (max(df.X)+min(df.Y))/2
    origin_y = (max(df.X)+min(df.Y))/2
    
    
    for idx in range(0,len(df)):
       x = df.X.iloc[idx]
       y = df.Y.iloc[idx]
       
       new_x = origin_x + math.cos(rad)*(x-origin_x)-math.sin(rad)*(x-origin_x)-min(df.X)
       new_y = origin_x + math.cos(rad)*(y-origin_y)+math.sin(rad)*(y-origin_y)-min(df.Y)
        
       tmp = pd.DataFrame([[new_x,new_y]])
       spin_df = spin_df.append(tmp)
       
    spin_df=spin_df.rename(columns=spin_df.iloc[0])
    spin_df = spin_df.iloc[1: , :]
       
       
       
       
    return spin_df
    

# In[sampling]
###in order to boost the number of data, sample with replacement,  data from the each seperated corrdinate data
##save the sampled data into csv for sake of meemory

def sample (all_elements,grouping_range,number_of_sample_from_one_img,sampliing_multiplier):
    begin_time = datetime.datetime.now()
    
    
    for k in range(0 , len(all_elements)):
        element_df_name = "line" + str(all_elements[k][0][0])+ "element" + str(all_elements[k][0][1])
        element_df = pd.DataFrame(all_elements[k][1][0])
        element_df.columns = ['X','Y']
        element_df.X = (element_df.X-min(element_df.X))*(100/(max(element_df.X)-min(element_df.X)))
        element_df.Y = (element_df.Y-min(element_df.Y))*(100/(max(element_df.Y)-min(element_df.Y)))        

        value = all_elements[k][0][1]
        sampling_size = int(len(element_df)*sampliing_multiplier)
        #sampling_size = int(len(element_df)*0.05)
        #sampling_size = len(element_df)
        for j in range(0,number_of_sample_from_one_img):
            
            random_rad = random.randrange(-5,5)
            
            #dat_sample =element_df
            dat_sample =element_df.sample(n=sampling_size)
            dat_sample = spin(random_rad, dat_sample)
            
            Xaxis = dat_sample.groupby(pd.cut(dat_sample["X"], np.arange(0,100,grouping_range))).size()/sampling_size
            Yaxis = dat_sample.groupby(pd.cut(dat_sample["Y"], np.arange(0,100,grouping_range))).size()/sampling_size
    
            Xaxis = Xaxis.to_frame()
            Yaxis = Yaxis.to_frame()
    
    
    
            # Xaxis = Xaxis.reset_index(level='X')
            # Yaxis = Yaxis.reset_index(level='Y')
            # Xaxis.columns = ['range','X']
            # Yaxis.columns = ['range','Y']
            Xaxis.columns = ['size']
            Yaxis.columns = ['size']
    
            x = Xaxis.transpose()
            y = Yaxis.transpose()
    
            xy = pd.concat([x,y],axis=1)
    
            #xy = xy*adj_val
    
            xy.columns = xy.columns.add_categories('val')
            xy['val'] = value
    
    
            names = pd.DataFrame([xy.columns.values]).transpose()        
            save(f'C:/Users/parth/Desktop/OCR/dataset/whole/{number_of_sample_from_one_img}/{sampliing_multiplier}/{element_df_name}'+'_xy.csv',xy,names)
            plt.scatter(dat_sample.X,dat_sample.Y)
    
    
    print("sampling data: " + str(datetime.datetime.now() - begin_time))
    
    
    
# In[things needed]:
##set to run all the functions created.
dat2= pd.read_csv('C:/Users/parth/Desktop/OCR/dataset/whole/data/dat2.csv')    
train_elements = splitter(line_dat=dat2, Y_range = 2, min_avg_x = 2)

test_data=pd.read_csv('C:/Users/parth/Desktop/OCR/dataset/whole/data/dat3.csv') 
test_elements = splitter(line_dat=test_data, Y_range = 2, min_avg_x = 2)

for k in range(0,10):
    element_df = pd.DataFrame(test_elements[k][1][0])
    element_df.columns = ['X','Y']
    element_df.X = (element_df.X-min(element_df.X))*(100/(max(element_df.X)-min(element_df.X)))
    element_df.Y = (element_df.Y-min(element_df.Y))*(100/(max(element_df.Y)-min(element_df.Y)))
    Xaxis = element_df.groupby(pd.cut(element_df["X"], np.arange(0,100,4))).size()/len(element_df)
    Yaxis = element_df.groupby(pd.cut(element_df["Y"], np.arange(0,100,4))).size()/len(element_df)
    Xaxis = Xaxis.to_frame()
    Yaxis = Yaxis.to_frame()
    Xaxis.columns = ['size']
    Yaxis.columns = ['size']
    x = Xaxis.transpose()
    y = Yaxis.transpose()
    xy_test = pd.concat([x,y],axis=1)
    # return(xy_test)
    # return(im)
    
    
def get_compiled_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
    model.add(tf.keras.Input(shape=(10,)))
    model.add(tf.keras.layers.Dense(32,activation = 'relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model



path = 'C:/Users/parth/Desktop/OCR/dataset/whole/'
dat = glob.glob(path + "*csv")
xy_whole = pd.read_csv(dat[0])

for i in range (1, len(dat)):
    xy_data =  pd.read_csv(dat[i])
    xy_whole = xy_whole.append(xy_data, ignore_index=True)


target = xy_whole.pop('val')

dataset = tf.data.Dataset.from_tensor_slices((xy_whole.values, target.values))
for feat, targ in dataset.take(500):
    print ('Features: {}, Target: {}'.format(feat, targ))



for k in range(0 , len(train_elements)):
    element_df_name = "line" + str(train_elements[k][0][0])+ "element" + str(train_elements[k][0][1])
    element_df = pd.DataFrame(train_elements[k][1][0])
    element_df.columns = ['X','Y']
    plt.scatter(element_df.X,element_df.Y)
   
    
multi = np.arange(0.04, 0.05, 0.01)
# In[loop for hyperparameter]:   
correct = np.array([0,1,2,3,4,5,6,7,8,9])
multi = np.arange(0.05,0.2,0.05)
result_sheet = pd.DataFrame()




for sampliing_multiplier in multi:
    for number_of_sample_from_one_img in range(240,300,10):

        sample(train_elements,4,number_of_sample_from_one_img,sampliing_multiplier)
        
for sampliing_multiplier in multi:
    for number_of_sample_from_one_img in range(100,300,10):        
        path = f'C:/Users/parth/Desktop/OCR/dataset/whole/{number_of_sample_from_one_img}/{sampliing_multiplier}/{element_df_name}'
        dat = glob.glob(path + "*csv")
        xy_whole = pd.read_csv(dat[0])

        for i in range (1, len(dat)):
            xy_data =  pd.read_csv(dat[i])
            xy_whole = xy_whole.append(xy_data, ignore_index=True)


        target = xy_whole.pop('val')

        dataset = tf.data.Dataset.from_tensor_slices((xy_whole.values, target.values))
        for feat, targ in dataset.take(500):
            print ('Features: {}, Target: {}'.format(feat, targ))
        
        
        train_dataset = dataset.shuffle(len(xy_whole)).batch(1)
        model = get_compiled_model()
        model.fit(train_dataset, epochs=5)

        result_data = []
        for k in range(0,10):
            element_df = pd.DataFrame(test_elements[k][1][0])
            element_df.columns = ['X','Y']
            element_df.X = (element_df.X-min(element_df.X))*(100/(max(element_df.X)-min(element_df.X)))
            element_df.Y = (element_df.Y-min(element_df.Y))*(100/(max(element_df.Y)-min(element_df.Y)))
            Xaxis = element_df.groupby(pd.cut(element_df["X"], np.arange(0,100,4))).size()/len(element_df)
            Yaxis = element_df.groupby(pd.cut(element_df["Y"], np.arange(0,100,4))).size()/len(element_df)
            Xaxis = Xaxis.to_frame()
            Yaxis = Yaxis.to_frame()
            Xaxis.columns = ['size']
            Yaxis.columns = ['size']
            x = Xaxis.transpose()
            y = Yaxis.transpose()
            xy_test = pd.concat([x,y],axis=1)
            # return(xy_test)
            # return(im)
            p=model.predict(xy_test)
            rval = np.argmax(p[0])
            print(rval)
            result_data.append(rval)
            result_dat = pd.DataFrame(result_data)
            # return(xy_test)
            # return(im)
    
        score = np.count_nonzero(correct==result_data)
        result_data = pd.DataFrame(result_data)
        tmp = pd.DataFrame([[number_of_sample_from_one_img,sampliing_multiplier,score]])
        result_sheet = result_sheet.append(tmp)
    

# In[]:
    
result_sheet2 = pd.DataFrame()
para = result_sheet[result_sheet['score'] >8]
    
for sampliing_multiplier in para.multiplier:
    for number_of_sample_from_one_img in para.sample_size:

        sample(train_elements,4,number_of_sample_from_one_img,sampliing_multiplier)
        
        train_dataset = dataset.shuffle(len(xy_whole)).batch(1)
        model = get_compiled_model()
        model.fit(train_dataset, epochs=10)

        result_data = []
        for k in range(0,10):
            element_df = pd.DataFrame(test_elements[k][1][0])
            element_df.columns = ['X','Y']
            element_df.X = (element_df.X-min(element_df.X))*(100/(max(element_df.X)-min(element_df.X)))
            element_df.Y = (element_df.Y-min(element_df.Y))*(100/(max(element_df.Y)-min(element_df.Y)))
            Xaxis = element_df.groupby(pd.cut(element_df["X"], np.arange(0,100,4))).size()/len(element_df)
            Yaxis = element_df.groupby(pd.cut(element_df["Y"], np.arange(0,100,4))).size()/len(element_df)
            Xaxis = Xaxis.to_frame()
            Yaxis = Yaxis.to_frame()
            Xaxis.columns = ['size']
            Yaxis.columns = ['size']
            x = Xaxis.transpose()
            y = Yaxis.transpose()
            xy_test = pd.concat([x,y],axis=1)
            # return(xy_test)
            # return(im)
            p=model.predict(xy_test)
            rval = np.argmax(p[0])
            result_data.append(rval)
            # return(xy_test)
            # return(im)
    
        score = np.count_nonzero(correct==result_data)
        result_data = pd.DataFrame(result_data)
        tmp = pd.DataFrame([[number_of_sample_from_one_img,sampliing_multiplier,score]])
        result_sheet2 = result_sheet2.append(tmp)
