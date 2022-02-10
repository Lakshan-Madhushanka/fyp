#this file is for training
from classification_models.models.resnet import preprocess_input
import tensorflow.keras as tk
from tensorflow import reduce_sum
import numpy as np
import datagen as p1
import model_with_segmentation_models as p2
import segmentation_models as sm

#some useful parameters
h = 256
w = 256
BATCH = 8
smooth = 1e-15
EPOCHS = 500
PATH = '/content/drive/MyDrive/datasets/fulljaw_dataset'#path to the main folder that contains image directory and mask directory
lr = 0.0001
backbone = 'efficientnetb1'

#loading raw datasets...
(train_x, train_y), (valid_x, valid_y) = p1.load_data(PATH)

#obtaining organized dataset
train_dataset = p1.tf_dataset(train_x, train_y, batch=BATCH)
valid_dataset = p1.tf_dataset(valid_x, valid_y, batch=BATCH)

# #preprocessing input to match with segmentation_models package
# train_dataset = sm.get_preprocessing(backbone)
# train_dataset_x = preprocess_input(train_dataset_x)
# valid_dataset_x = preprocess_input(valid_dataset_x)
# #the only problem is, we cannot seperate out x and y parts of the dataset above. they have formed a dataframe that cannot be broken..
# #from the experiments I figured out, the preprocessing of train, val images isn't a must.

#importing custom model
custom_model = p2.custom_segmentation_model(w, h)

# #other compiling stuff
# def dice_coef(y_true, y_pred):
#     y_true = tk.layers.Flatten()(y_true)
#     y_pred = tk.layers.Flatten()(y_pred)
#     intersection = reduce_sum(y_true*y_pred)
#     dc = (2*intersection + smooth)/(reduce_sum(y_true)+reduce_sum(y_pred)+smooth)
#     return dc
    
# def dice_loss(y_true, y_pred):
#     return 1.0-dice_coef(y_true, y_pred)

#training and saving
opt = tk.optimizers.Adam(lr)
metrics = [sm.metrics.iou_score, tk.metrics.Recall(), tk.metrics.Precision()]
custom_model.compile(loss=sm.losses.bce_jaccard_loss, optimizer=opt, metrics=metrics)

callbacks = [
    tk.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.90, patience=10),
    tk.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=False),
    tk.callbacks.ModelCheckpoint('bestModel_fulljaw.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto', period=2)
]

train_steps = len(train_x)//BATCH
valid_steps = len(valid_x)//BATCH

if len(train_x) % BATCH != 0:
    train_steps += 1
    
if len(valid_x) % BATCH != 0:
    valid_steps += 1
    

custom_model.fit(
    train_dataset,
    validation_data = valid_dataset,
    epochs = EPOCHS,
    steps_per_epoch = train_steps,
    validation_steps = valid_steps,
    callbacks = callbacks
)

custom_model.save('models/model_fulljaws1.h5')
