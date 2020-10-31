# skin-lesion-using-ensemble-learning
Deep Ensemble Learning for Skin Lesion Classification

Skin cancer many death every year globally. If it is diagnosed in early stage it is curable
. Auto classification of the different skin lesion is a tough task to provide clinicians with the ability to differentiate between all different kind of lesion categories and recommend the best and the suitable treatment. This project focuses on the identification of this disease using ensemble learning of state-of-the-art deep learning approaches. Since our framework is realized within a single neural net architecture, all the parameters of the member CNNs and the weights applied in the fusion can be determined by backpropagation routinely applied for such tasks
Our main aim is to develop such an automated framework that efficiently performs a reliable automatic lesion classification to seven skin lesion types. In this task, we propose a deep neural network based framework which follows an ensemble approach with the help of pre trained neural network.
Keywords--skin lesions, diagnosis, melanoma, deep neural networks, deep learning.


#CODE
import os
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle as pkl
import ISIC_dataset as ISIC
from metrics import dice_loss, jacc_loss, jacc_coef, dice_jacc_mean

np.random.seed(4)
K.set_image_dim_ordering('th')  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow
# Lesion Segmentation: Training Image and Mask
training_folder = "datasets/ISIC-2017_Training_Data"
training_mask_folder = "datasets/ISIC-2017_Training_Part1_GroundTruth"

# Lesion Classification: Training Labels
training_labels_csv = "datasets/ISIC-2017_Training_Part3_GroundTruth.csv"

# Lesion Segmentation: Validation Image
validation_folder = "datasets/ISIC-2017_Validation_Data"
# Lesion Segmentation: Test Image
test_folder = "datasets/ISIC-2017_Test_v2_Data"
# Resize image dimension
height, width = 128, 128

# Image parameters
mean_type = 'imagenet' # 'sample' 'samplewise'
rescale_mask = True
use_hsv = False
dataset = 'isic' # 'isic' 'isicfull' 'isic_noval_notest' 'isic_other_split' 'isic_notest'

# Model parameters
model_name = "model1"
seed = 1
nb_epoch = 1  # 220
initial_epoch = 0 
batch_size = 4
loss_param = 'dice'
optimizer_param = 'adam'
monitor_metric = 'val_jacc_coef'
fc_size = 8192

# Run-time flags
do_train = True # train network and save as model_name
do_predict = True # use model to predict and save generated masks for Validation/Test
do_ensemble = False # use previously saved predicted masks from multiple models to generate final masks
ensemble_pkl_filenames = ["model1","model2", "model3", "model4"]

# Training metric options
metrics = [jacc_coef]

# HSV options: On or off
if use_hsv:
    n_channels = 6
    print("Using HSV")
else:
    n_channels = 3


# Image mean options
print(("Using {} mean".format(mean_type)))

remove_mean_imagenet   = False
remove_mean_samplewise = False
remove_mean_dataset    = False

if mean_type == 'imagenet':
    remove_mean_imagenet = True
    
elif mean_type == 'sample':
    remove_mean_samplewise = True
    
elif mean_type == 'dataset':
    remove_mean_dataset = True
    train_mean = np.array([[[ 180.71656799]],[[ 151.13494873]],[[ 139.89967346]]]);
    train_std = np.array([[[1]],[[1]],[[ 1]]]); # not using std

else:
    raise Exception("Wrong mean type")
    

# Loss options    
loss_options = {'BCE': 'binary_crossentropy', 'dice':dice_loss, 'jacc':jacc_loss, 'mse':'mean_squared_error'}
loss = loss_options[loss_param]


# Optimizer options
optimizer_options = {'adam': Adam(lr=1e-5),
                     'sgd': SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)}
optimizer = optimizer_options[optimizer_param]


# Specify model filename
model_filename = "weights/{}.h5".format(model_name)


print('Create model')


import numpy as np
from keras.models import Model
from keras.layers import merge, Flatten, Dense, Input, Dropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers.noise import GaussianNoise

from keras.layers import concatenate

import h5py
np.random.seed(4)

# VGG16_WEIGHTS_NOTOP = 'pretrained_weights/vgg16_notop.h5'
VGG16_WEIGHTS_NOTOP = 'pretrained_weights/vgg16_weights.h5'
# download .h5 weights from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

def Unet3(img_rows, img_cols, loss , optimizer, metrics, fc_size = 8192, channels = 3):
    filter_size = 5
    filter_size_2 = 11
    dropout_a = 0.5
    dropout_b = 0.5
    dropout_c = 0.5
    gaussian_noise_std = 0.025

    inputs = Input((channels, img_rows, img_cols))
    input_with_noise = GaussianNoise(gaussian_noise_std)(inputs)

    conv1 = Conv2D(32, (filter_size, filter_size), activation='relu', padding='same')(input_with_noise)
    conv1 = Conv2D(32, (filter_size, filter_size), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, (filter_size, filter_size), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    pool1 = GaussianNoise(gaussian_noise_std)(pool1)

    conv2 = Conv2D(64, (filter_size, filter_size), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (filter_size, filter_size), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, (filter_size, filter_size), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    pool2 = GaussianNoise(gaussian_noise_std)(pool2)

    conv3 = Conv2D(128, (filter_size, filter_size), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (filter_size, filter_size), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, (filter_size, filter_size), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    pool3 = Dropout(dropout_a)(pool3)

    convn = Conv2D(256, (filter_size, filter_size), activation='relu', padding='same')(pool3)
    convn = Conv2D(256, (filter_size, filter_size), activation='relu', padding='same')(convn)
    convn = Conv2D(256, (filter_size, filter_size), activation='relu', padding='same')(convn)
    pooln = MaxPooling2D((2, 2), strides=(2, 2))(convn)
    pooln = Dropout(dropout_a)(pooln)

    fc = Flatten()(pooln)
    fc = Dense(4096, activation='relu')(fc)
    fc = Dropout(dropout_b)(fc)

    n = img_rows / 2 / 2 / 2 / 2
    fc = Dense(int(256 * n * n), activation='relu')(fc)
    fc = GaussianNoise(gaussian_noise_std)(fc)
    fc = Reshape((256, int(n), int(n)))(fc)

    up0 = concatenate([UpSampling2D(size=(2, 2))(fc), convn], axis=1)
    up0 = Dropout(dropout_c)(up0)

    convp = Conv2D(256, (filter_size_2, filter_size_2), activation='relu', padding='same')(up0)
    convp = Conv2D(256, (filter_size, filter_size), activation='relu', padding='same')(convp)
    convp = Conv2D(128, (filter_size, filter_size), activation='relu', padding='same')(convp)

    up1 = concatenate([UpSampling2D(size=(2, 2))(convp), conv3], axis=1)
    up1 = Dropout(dropout_c)(up1)

    conv4 = Conv2D(128, (filter_size_2, filter_size_2), activation='relu', padding='same')(up1)
    conv4 = Conv2D(128, (filter_size, filter_size), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(64, (filter_size, filter_size), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv2], axis=1)
    up2 = Dropout(dropout_c)(up2)

    conv5 = Conv2D(64, (filter_size_2, filter_size_2), activation='relu', padding='same')(up2)
    conv5 = Conv2D(64, (filter_size, filter_size), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(32, (filter_size, filter_size), activation='relu', padding='same')(conv5)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv1], axis=1)
    up3 = Dropout(dropout_c)(up3)

    conv6 = Conv2D(32, (filter_size_2, filter_size_2), activation='relu', padding='same')(up3)
    conv6 = Conv2D(32, (filter_size, filter_size), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(32, (filter_size, filter_size), activation='relu', padding='same')(conv6)

    conv7 = Conv2D(1, (1, 1), activation='sigmoid')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def VGG16(img_rows, img_cols, pretrained, freeze_pretrained, loss , optimizer, metrics, channels=3):
    inputs = Input((channels, img_rows, img_cols))
    
    pad1 = ZeroPadding2D((1, 1), input_shape=(channels, img_rows, img_cols))(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(pad1)
    conv1 = ZeroPadding2D((1, 1))(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    pad2 = ZeroPadding2D((1, 1))(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(pad2)
    conv2 = ZeroPadding2D((1, 1))(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    pad3 = ZeroPadding2D((1, 1))(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(pad3)
    conv3 = ZeroPadding2D((1, 1))(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(conv3)
    conv3 = ZeroPadding2D((1, 1))(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    pad4 = ZeroPadding2D((1, 1))(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(pad4)
    conv4 = ZeroPadding2D((1, 1))(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(conv4)
    conv4 = ZeroPadding2D((1, 1))(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    pad5 = ZeroPadding2D((1, 1))(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(pad5)
    conv5 = ZeroPadding2D((1, 1))(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(conv5)
    conv5 = ZeroPadding2D((1, 1))(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(conv5)

    # Additional seven-layers just for loading weights from VGG-16
    pool_a = MaxPooling2D((2, 2), strides=(2, 2))(conv5)
    flat_a = Flatten()(pool_a)
    dense_a = Dense(4096, activation='relu')(flat_a)
    dense_a = Dropout(0.5)(dense_a)
    dense_b = Dense(4096, activation='relu')(dense_a)
    dense_b = Dropout(0.5)(dense_b) 
    dense_c = Dense(1000, activation='softmax')(dense_b)
    
    model = Model(inputs=inputs, outputs=dense_c)
    
    # Load weights
    if pretrained:
        weights_path = VGG16_WEIGHTS_NOTOP
        model.load_weights(weights_path, by_name=True)
        
        if freeze_pretrained:
            for layer in model.layers:
                layer.trainable = False
    
    # Remove the last seven-layers
    for i in range(7):
        model.layers.pop()
    
    dropout_val = 0.5
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    up6 = Dropout(dropout_val)(up6)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    up7 = Dropout(dropout_val)(up7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    up8 = Dropout(dropout_val)(up8)

    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    up9 = Dropout(dropout_val)(up9)

    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
    
    
    
    # import models
model = 'vgg'

if model == 'unet':
    model = Unet(height, width,
                 loss=loss,
                 optimizer=optimizer,
                 metrics=metrics,
                 fc_size=fc_size,
                 channels=n_channels)

elif model == 'unet2':
    model = Unet2(height, width,
                  loss=loss,
                  optimizer=optimizer,
                  metrics=metrics,
                  fc_size=fc_size,
                  channels=n_channels)

elif model == 'unet3':
    model = Unet3(height, width,
                  loss=loss,
                  optimizer=optimizer,
                  metrics=metrics,
                  fc_size=fc_size,
                  channels=n_channels)

elif model == 'vgg':
    model = VGG16(height, width,
                  pretrained=True,
                  freeze_pretrained=False,
                  loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
else:
    print("Incorrect model name")
    
    
    
    
    
    
    
    if do_train:
    if dataset == 'isicfull':
        n_samples = 2000 # per epoch
        print("Using ISIC full dataset")
        
        # Get train, validation and test image list
        train_list, val_list, test_list = ISIC.train_val_test_from_txt(isicfull_train_split, 
                                                                       isicfull_val_split, 
                                                                       isicfull_test_split)
        
        # Folder for resized images
        base_folder = "datasets/isicfull_{}_{}".format(height,width)
        image_folder = os.path.join(base_folder,"image")
        mask_folder = os.path.join(base_folder,"mask")
        
        # Create folder and generate resized images if folder does not exists
        if not os.path.exists(base_folder):  
            print("Begin resizing...")
            
            ISIC.resize_images(train_list+val_list+test_list, 
                               input_image_folder = isicfull_folder,
                               input_mask_folder = isicfull_mask_folder, 
                               output_image_folder = image_folder.format(height,width), 
                               output_mask_folder = mask_folder, 
                               height = height, 
                               width = width)
            
            print("Done resizing...")
            
    else:
        print("Using ISIC 2017 dataset")
        
        # Folders for resized images
        base_folder = "datasets/isic_{}_{}".format(height, width)
        image_folder = os.path.join(base_folder, "image")
        mask_folder = os.path.join(base_folder, "mask")

        # train_list, train_label, test_list, test_label = ISIC.train_test_from_yaml(yaml_file = training_split_yml, csv_file = training_labels_csv)
        
        # Get train, validation and test image list based on training dataset
        df = pd.read_csv(training_labels_csv)
        df['image_id'] = df['image_id'].astype(str) + '.jpg'
        train_list, test_list, train_label, test_label = ISIC.train_val_split(df['image_id'].tolist(), df['melanoma'].tolist(), seed = seed, val_split = 0.20)
        train_list, val_list, train_label, val_label = ISIC.train_val_split(train_list, train_label, seed = seed, val_split = 0.20)
        
        # Create folder and generate resized images if folder does not exists
        if not os.path.exists(base_folder):
            ISIC.resize_images(train_list+val_list+test_list,
                               input_image_folder = training_folder, 
                               input_mask_folder = training_mask_folder, 
                               output_image_folder = image_folder, 
                               output_mask_folder = mask_folder, 
                               height = height, 
                               width = width)
            
        if dataset == "isic_notest": # previous validation split will be used for training
            train_list = train_list + val_list
            val_list = test_list
            
        elif dataset =="isic_noval_notest": # previous validation/test splits will be used for training
            monitor_metric = 'jacc_coef'
            train_list = train_list + val_list + test_list 
            val_list = test_list
            
        elif dataset =="isic_other_split": # different split, uses previous val/test for training
            seed = 82
            train_list1, train_list2, train_label1, train_label2 = ISIC.train_val_split(train_list, train_label, seed=seed, val_split=0.30)
            train_list = val_list+test_list+train_list1 
            val_list = train_list2
            test_list = val_list 
            
        n_samples = len(train_list)
        # n_samples = 20
        
    print("Loading images")
    # Assign train, validation and test image and mask based on training dataset 
    train, train_mask = ISIC.load_images(train_list, 
                                         height, width, 
                                         image_folder, mask_folder,
                                         remove_mean_imagenet = remove_mean_imagenet,
                                         rescale_mask = rescale_mask, 
                                         use_hsv = use_hsv, 
                                         remove_mean_samplewise = remove_mean_samplewise)
    
    val, val_mask = ISIC.load_images(val_list, height, width, 
                                     image_folder, mask_folder,  
                                     remove_mean_imagenet = remove_mean_imagenet, 
                                     rescale_mask = rescale_mask, 
                                     use_hsv = use_hsv, 
                                     remove_mean_samplewise = remove_mean_samplewise)
    
    test, test_mask = ISIC.load_images(test_list, height, width, 
                                       image_folder, mask_folder,
                                       remove_mean_imagenet = remove_mean_imagenet, 
                                       rescale_mask = rescale_mask, 
                                       use_hsv = use_hsv, 
                                       remove_mean_samplewise = remove_mean_samplewise)
    print("Done loading images")
    
    # Remove mean of train, val and test images
    if remove_mean_dataset:  # Only True when specify mean_type = 'dataset'
        print(("\nUsing Train Mean: {} Std: {}".format(train_mean, train_std)))
        train = (train - train_mean)/train_std
        val   = (val - train_mean)/train_std
        test  = (test - train_mean)/train_std

    # Batch size 
    print(("Using batch size = {}".format(batch_size)))
    
    print('Fit model')
    # Save best model
    model_checkpoint = ModelCheckpoint(model_filename, monitor=monitor_metric, save_best_only=True, verbose=1)
    
    # Define dictionary for data augmentation
    data_gen_args = dict(featurewise_center = False, 
                         samplewise_center = remove_mean_samplewise,
                         featurewise_std_normalization = False, 
                         samplewise_std_normalization = False, 
                         zca_whitening = False, 
                         rotation_range = 270, 
                         width_shift_range = 0.1, 
                         height_shift_range = 0.1, 
                         horizontal_flip = False, 
                         vertical_flip = False, 
                         zoom_range = 0.2,
                         channel_shift_range = 0,
                         fill_mode = 'reflect',
                         dim_ordering = K.image_dim_ordering())
    data_gen_mask_args = dict(list(data_gen_args.items()) + list({'fill_mode':'nearest','samplewise_center':False}.items()))
    
    # Perform data augmentation using Keras ImageDataGenerator
    print("Create Data Generator")
    train_datagen = ImageDataGenerator(data_gen_args)
    train_mask_datagen = ImageDataGenerator(data_gen_mask_args)
    train_generator = train_datagen.flow(train, batch_size=batch_size, seed=seed)
    train_mask_generator = train_mask_datagen.flow(train_mask, batch_size=batch_size, seed=seed)
    train_generator_f = myGenerator(train_generator, train_mask_generator, remove_mean_imagenet=remove_mean_imagenet, rescale_mask=rescale_mask, use_hsv=use_hsv)
    
    # Train model using train list and validate using val list
    if dataset == "isic_noval_notest":
        print("Not using validation during training")
        history = model.fit_generator(train_generator_f,
                                      # samples_per_epoch=n_samples,
                                      steps_per_epoch = n_samples,
                                      nb_epoch = nb_epoch,
                                      callbacks = [model_checkpoint], 
                                      initial_epoch = initial_epoch)
    else:  # default model fitting
        model.load_weights(model_filename)
        history = model.fit_generator(train_generator_f,
                                      # samples_per_epoch=n_samples,
                                      steps_per_epoch = n_samples,
                                      nb_epoch=nb_epoch, 
                                      validation_data=(val,val_mask), 
                                      callbacks=[model_checkpoint], 
                                      initial_epoch=initial_epoch)

    train = None
    train_mask = None # clear memory
    
    # Load best saved checkpoint after training
    print("Load best checkpoint")
    model.load_weights(model_filename) 

    # Evaluate model using val list and test list aka subset of training dataset
    mask_pred_val = model.predict(val) 
    mask_pred_test = model.predict(test)
    
    for pixel_threshold in [0.5]: #np.arange(0.3,1,0.05):
        # Predict mask for val list
        mask_pred_val = np.where(mask_pred_val>=pixel_threshold, 1, 0)  # assign pixel 0/1 based on output layer activation value
        mask_pred_val = mask_pred_val * 255  # assign 0 -> 0 and 1 -> 255
        mask_pred_val = mask_pred_val.astype(np.uint8)
        print("Validation Predictions Max: {}, Min: {}".format(np.max(mask_pred_val), np.min(mask_pred_val)))
        
        # Evaluate Jaccard score for val
        print(model.evaluate(val, val_mask, batch_size = batch_size, verbose=1))
        dice, jacc = dice_jacc_mean(val_mask, mask_pred_val, smooth = 0)
        print(model_filename)
        print("Resized val dice coef      : {:.4f}".format(dice))
        print("Resized val jacc coef      : {:.4f}".format(jacc))

        # Predict mask for test list
        mask_pred_test = np.where(mask_pred_test>=pixel_threshold, 1, 0)
        mask_pred_test = mask_pred_test * 255
        mask_pred_test = mask_pred_test.astype(np.uint8)
        
        # Evaluate Jaccard score for test
        print(model.evaluate(test, test_mask, batch_size = batch_size, verbose=1))
        dice, jacc = dice_jacc_mean(test_mask, mask_pred_test, smooth = 0)
        print("Resized test dice coef      : {:.4f}".format(dice))
        print("Resized test jacc coef      : {:.4f}".format(jacc))
else:
    # Load model directly when do_train=False
    print('Load model')
    model.load_weights(model_filename)
    
    
    
def predict_challenge(challenge_folder, challenge_predicted_folder, mask_pred_challenge=None, plot=True):
    # Get challenge folder and create new folder with resized images
    challenge_list = ISIC.list_from_folder(challenge_folder)
    challenge_resized_folder = challenge_folder+"_{}_{}".format(height,width)
    
    if not os.path.exists(challenge_resized_folder):
        ISIC.resize_images(challenge_list, 
                           input_image_folder=challenge_folder, 
                           input_mask_folder=None, 
                           output_image_folder=challenge_resized_folder, 
                           output_mask_folder=None, 
                           height=height, 
                           width=width)

    challenge_resized_list =  [name.split(".")[0]+".png" for name in challenge_list]
    challenge_images = ISIC.load_images(challenge_resized_list, 
                                        height, width, image_folder=challenge_resized_folder,
                                        mask_folder=None, 
                                        remove_mean_imagenet=True, 
                                        use_hsv = use_hsv,
                                        remove_mean_samplewise=remove_mean_samplewise)
    
    # Remove image mean from dataset
    if remove_mean_dataset:
        challenge_images = (challenge_images-train_mean)/train_std
    if mask_pred_challenge is None:
        mask_pred_challenge = model.predict(challenge_images)
        
    with open('{}.pkl'.format(os.path.join(challenge_predicted_folder,model_name)), 'wb') as f:
        pkl.dump(mask_pred_challenge, f)
        
    # Create mask prediction for challenge images
    mask_pred_challenge = np.where(mask_pred_challenge>=0.5, 1, 0)
    mask_pred_challenge = mask_pred_challenge * 255
    mask_pred_challenge = mask_pred_challenge.astype(np.uint8)

    challenge_predicted_folder = os.path.join(challenge_predicted_folder, model_name)
    if not os.path.exists(challenge_predicted_folder):
        os.makedirs(challenge_predicted_folder)

    print("Start challenge prediction:")
    for i in range(len(challenge_list)):
        print(("{}: {}".format(i, challenge_list[i])))
        # Revert predicted mask to original image resolution
        ISIC.show_images_full_sized(image_list = challenge_list, 
                                    img_mask_pred_array = mask_pred_challenge, 
                                    image_folder=challenge_folder, 
                                    mask_folder=None, 
                                    index = i, 
                                    output_folder=challenge_predicted_folder, 
                                    plot=plot)
        
        def join_predictions(pkl_folder, pkl_files, binary=False, threshold=0.5):
    n_pkl = float(len(pkl_files))
    array = None
    for fname in pkl_files:
        with open(os.path.join(pkl_folder,fname+".pkl"), "rb") as f:
            tmp = pkl.load(f)
            if binary:
                tmp = np.where(tmp>=threshold, 1, 0)
            if array is None:
                array = tmp
            else:
                array = array + tmp
    return array/n_pkl
    
    
    
    
    validation_folder = 'datasets/ISIC-2017_Validation_Data'
validation_predicted_folder = 'results/ISIC-2017_Validation_Predicted'












if do_predict:
    # free memory
    train = None
    train_mask = None
    val = None
    test = None 
    
    print("Start Challenge Validation")
    predict_challenge(challenge_folder=validation_folder, challenge_predicted_folder=validation_predicted_folder, plot=False)
    
    
    
    
    
    import numpy as np
import pandas as pd
import os
import cv2
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score


# Lesion Classification: Training Labels
training_labels_csv = "datasets/ISIC-2017_Training_Part3_GroundTruth.csv"
validation_labels_csv = "datasets/ISIC-2017_Validation_Part3_GroundTruth.csv"
test_labels_csv = "datasets/ISIC-2017_Test_v2_Part3_GroundTruth.csv"

# Lesion Segmentation: Training Image and Mask
training_folder = "datasets/ISIC-2017_Training_Data"
training_mask_folder = "datasets/ISIC-2017_Training_Part1_GroundTruth"
# Lesion Segmentation: Validation Image
validation_folder = "datasets/ISIC-2017_Validation_Data"
validation_mask_folder = "datasets/ISIC-2017_Validation_Part1_GroundTruth/"
validation_pred_folder = "results/ISIC-2017_Validation_Predicted/model1/"
# Lesion Segmentation: Test Image
test_folder = "datasets/ISIC-2017_Test_v2_Data"
test_mask_folder = "datasets/ISIC-2017_Test_v2_Part1_GroundTruth/"
test_pred_folder = "results/ISIC-2017_Test_v2_Predicted/model1/"


smooth_default = 1.

def dice_coef(y_true, y_pred, smooth = smooth_default, per_batch = True):
    if not per_batch:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    else: 
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersec = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
        union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
        return K.mean(intersec / union)
    
def jacc_coef(y_true, y_pred, smooth = smooth_default):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    
def jacc_loss(y_true, y_pred):
    return -jacc_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    
def dice_jacc_single(mask_true, mask_pred, smooth = smooth_default):
    bool_true = mask_true.reshape(-1).astype(np.bool)
    bool_pred = mask_pred.reshape(-1).astype(np.bool)
    if bool_true.shape != bool_pred.shape:
        raise ValueError("Masks of different sizes.")

    bool_sum = bool_true.sum() + bool_pred.sum()
    if bool_sum == 0:
        print("Empty mask")
        return 0,0
    intersec = np.logical_and(bool_true, bool_pred).sum()
    dice = 2. * intersec / bool_sum
    jacc = jaccard_similarity_score(bool_true.reshape((1, -1)), bool_pred.reshape((1, -1)), normalize=True, sample_weight=None)
    return dice, jacc

def dice_jacc_mean(mask_true, mask_pred, smooth = smooth_default):
    dice = 0
    jacc = 0
    for i in range(mask_true.shape[0]):
        current_dice, current_jacc = dice_jacc_single(mask_true=mask_true[i],mask_pred=mask_pred[i], smooth= smooth)
        dice = dice + current_dice
        jacc = jacc + current_jacc
    return dice/mask_true.shape[0], jacc/mask_true.shape[0]

def list_from_folder(image_folder):
    image_list = []
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".png"):
            image_list.append(image_filename)
    print(("Found {} images.".format(len(image_list))))
    return image_list


print("Calculating Jaccard Similarity Score for Validation Set")
val_mask_list = list_from_folder(validation_mask_folder)
df_val = pd.read_csv(validation_labels_csv)
jacc_val_list = []
dice_val_list = []
for i in range(len(val_mask_list)):
    print(str(i)+': '+str(val_mask_list[i]))
    mask_true = cv2.imread(validation_mask_folder+str(val_mask_list[i]))
    mask_pred = cv2.imread(validation_pred_folder+str(val_mask_list[i]))
    dice, jacc = dice_jacc_single(mask_true=mask_true, mask_pred=mask_pred)
    jacc_val_list.append(jacc)
    dice_val_list.append(dice)
df_val['jacc'] = jacc_val_list
df_val['dice'] = dice_val_list
print(df_val.head())
print('Average Jaccard Score = '+str(np.mean(jacc_val_list)))
print('Average Dice coefficient = '+str(np.mean(dice_val_list)))
df_val.to_csv('val.csv', encoding='utf-8', index=False)


                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                

