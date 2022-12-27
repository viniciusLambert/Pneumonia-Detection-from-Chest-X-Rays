# FDA  Submission

**Your Name:** Vinicius N

**Name of your Device:** Deep Learning Model for Pneumonia Detection on Chest X-Rays

## Algorithm Description 

### 1. General Information


**Intended Use Statement:** 

    The algorithm uses CNNs to analyze Chest X-rays images and find pneumonia presence. It's intent to help radiologists in a clinical setting and should be used ion conjunction with a Radiologist review.

**Indications for Use:**
    Use it in patients whose medical history indicates pneumonia infection, of ages between 1 and 95. X-Ray images must be taken with a PA or AP viewing position.

**Device Limitations:**
    It may be used in a computer that meets the minimum GPU and RAM requirements

**Clinical Impact of Performance:**
    When the model incorrectly predicts a False Positive, a patient may be directed to a doctor and the diagnosis can be validated with a new diagnosis.

    False negatives can impact a patient. If the false negative is not revised by a radiologist, the case can get worse, impacting putting in risk the patient's life.

    In all cases, the Radiologist review is critical.

### 2. Algorithm Design and Function



**DICOM Checking Steps:**
<br>Check DICOM Headers for: 
<br>Modality == 'DX'
<br>BodyPartExamined=='CHEST'
<br>PatientPosition in 'PA' or'AP' Position
<br>If any of these three categories do not match their respective requirements, 
then a message will state that the DICOM does not meet criteria. 

**Preprocessing Steps:**
    For DICOMs that pass the intial Header check, the DICOM pixel array will be opened. A copy of the DICOM pixel array will be normalized and then resized to 224 by 224 pixels.
**CNN Architecture:**

A Sequential Model was built by Fine-tuning the VGG16 Model with ImageNet weights.
<br>This model takes the VGG16 model layers up to and including the block5_pool.  <br>Layers taken from the VGG16 model were frozen, so that their weights were not trained.  Output from this pre-trained model was flattened.

<br>These following layers were added:
<br>(Dense(2048, activation = 'relu'))
<br>(Dropout(0.5))
<br>(Dense(1024, activation = 'relu'))
<br>(Dropout(0.5))
<br>(Dense(512, activation = 'relu'))
<br>(Dropout(0.5))
<br>(Dense(256, activation = 'relu'))
<br>(Dense(1, activation = 'sigmoid'))
<br>Output


### 3. Algorithm Training

**Parameters:**
Keras.preprocessing.image ImageDataGenerator was used with the following parameters to augment training images.    
train_gen = idg.flow_from_dataframe(dataframe=train_df, 
                                         directory=None, 
                                         x_col = 'path',
                                         y_col = 'pneumonia_class',
                                         class_mode = 'binary',
                                         target_size = IMG_SIZE, 
                                         batch_size = 64)

The Image Data Generator was applied to the training data.  The training data was resized to 224x224 and divided into batch size of 16.  

val_gen = val_idg.flow_from_dataframe(dataframe=valid_df, 
                                         directory=None, 
                                         x_col = 'path',
                                         y_col = 'pneumonia_class',
                                         class_mode = "binary",
                                         target_size = IMG_SIZE, 
                                         batch_size = 128)



The optimizer Adam was applied to the training dataset with a learning rate of 0.001.  Binary Cross Entropy was the loss parameter.

optimizer = Adam(learning_rate = 1e-3)
loss = 'binary_crossentropy'
metrics = ['binary_accuracy']

<< Insert algorithm training performance visualization >> 

<< Insert P-R curve >>

**Final Threshold and Explanation:**

<br>For this project, the first model is the best architecture and its optimal threshold value is 0.161 as determined from F1. This combination yields a F1 score of 0.335.

### 4. Databases
 The Data_Entr_2017.cvs contains 1,431 Images positive for Pneumonia and 110,689 Images negative for Pneumonia.

The data set was split into 80% Training data and 20% for validation data.

**Description of Training Dataset:** 
    For the training data set, the positive to negative images must be equal in number.
Training DataSet: Pneumonia Positive (80% of all Pneumonia Positive cases in dataset) = 1145 counts, Pneumonia Negative = 1145 counts

**Description of Validation Dataset:** 
    For the validation dataset, the positive and negative images are distributed like the real world, with 20% of the images containing positive cases.
Validation DataSet: Pneumonia Positive (20% of all Pneumonia Positive cases in dataset) = 286 counts, Pneumonia Negative = 1144 counts

### 5. Ground Truth

The dataset used in this project was curated by the NIH. It is comprised of 112,120 X-Ray images with disease labels from 30,805 unique patients. The disease labels for each image were created using Natural Language Processing (NLP) to process associated radiological reports. The estimated accuracy of the NLP labeling accuracy is estimated to be >90%.

This limitation on the NLP accuracy cannot be rectified, because the original radiology reports are not publically available to review and confirm labels. If these reports are available for human review, then the labels' accuracy will be reasonably higher.

### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**
The patient population should be approximately a 1:1 ratio of males to females aged 95 years and younger and screened for pneumonia. The expected rate of positive cases of pneumonia is approximately 20% of those screened. The imaging modality must be X-rays (code DX) from frontal positions (PA or AP) on the patient's chest.
**Ground Truth Acquisition Methodology:**
The purpose of this device is to support the radiologist in his workflow. The optimal ground truth for this purpose would be the radiologist's review of the x-ray with the medical history available to him, including laboratory reports. To circumvent inaccuracies introduced by NLP, a digital form should be provided to radiologists to mark diseases they see on each X-ray exam. The checklist would include common lung diseases like the 14 diseases in this dataset.
**Algorithm Performance Standard:**
Based on the provided paper by P. Rajpurkarpar, et al., the standard of performance is F1 scores comparing radiologists and algorithms. F1 scores are the harmonic average of the precision and accessibility of the models.
Rajpurkarpar's CheXNet algorithm achieves an F1 score of 0.435, while radiologists achieved an F1 score of 0.387. These would be the benchmarks against which newly developed algorithms can be compared.