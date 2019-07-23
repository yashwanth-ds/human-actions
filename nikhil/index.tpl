


% from keras.models import load_model
% from keras.models import Sequential
% from keras.preprocessing.image import ImageDataGenerator
% import cv2
% import numpy as np
% from keras.optimizers import SGD
% from tqdm import tqdm
% import matplotlib.pyplot as plt
% import os
% from random import shuffle
% #model = load_model('model-accuracy-99%.h5')
% test_dir = r'E:\Nikhil\python\videoclass\videoclass\test'
% test1_dir = r'E:\Nikhil\python\videoclass\videoclass\test\test1'
% test2_dir=r'E:\Nikhil\python\machinelearningex\videoclass\videoclass\test'
% train1_data=r'E:\Nikhil\python\machinelearningex\videoclass\videoclass\train'
% validation1_data=r'E:\\Nikhil\\python\\machinelearningex\\videoclass\\videoclass\\test\\'
% model=Sequential()
% model=load_model('model4-imggen.h5')
% lrate=0.01
% epochs1=50
% decay=lrate/epochs1
% sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)

% model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
% test_datagen= ImageDataGenerator(rescale=1./255)

%test_generator = test_datagen.flow_from_directory(
%     name,
%     target_size=(120,120),
%     classes=[name1],
%     shuffle=False,
%     batch_size=name2,
%     )
% train_datagen = ImageDataGenerator(
%     rescale=1. / 255,
%     shear_range=0.2,
%     zoom_range=0.2,
%     horizontal_flip=True)
% train_generator = train_datagen.flow_from_directory(
%     train1_data,
%     target_size=(120,120),
%     batch_size=12,
%     shuffle=False,
%     classes=['boxing1','handclapping','handwaving','walking'],
%     )
% test_generator.reset()
% train_generator.reset()
% #print(test_generator.filenames)
% #model1=model.predict_generator(test_generator,1200)
% test_datagen = ImageDataGenerator(rescale=1. / 255)
% validation_generator = test_datagen.flow_from_directory(
%    validation1_data,
%     target_size=(120,120),
%    classes=['boxing','handclapping','handwaving','walking'],
%    batch_size=12,
%    )




% pred= model.predict_generator(test_generator, 1)
% predicted_class_indices=np.argmax(pred,axis=1)
% labels = (validation_generator.class_indices)
% labels2 = dict((v,k) for k,v in labels.items())
% predictions = [labels2[k] for k in predicted_class_indices]
% #print(predicted_class_indices)
% #print (labels)
        
% #print (predictions)

% dt={}


% for i in range(len(test_generator.filenames)):
    
%     	name=test_generator.filenames[i].split('\\')[-1]
%     	dt[name]=predictions[i]
%	print(dt)
% dt1=list(dt.values())

<h1>Image Name:  {{name}}</h1>
		
%	
% #dt=dt1()

% #print(len(dt))



<div class="row">
	<div class="column">
		<img src="/images/{{name1}}/{{name}}">
</div>
</div>
<h1>the predicted action in the output Image {{name}} is {{dt1[i]}}</h1>
