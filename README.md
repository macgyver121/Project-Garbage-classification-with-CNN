# Introduction

จุดประสงค์การศึกษาครั้งนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep learning คณะสถิติประยุกต์ สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล(DADS) สถาบันบัณฑิตพัฒนบริหารศาสตร์ โดยมี ผศ.ดร.ฐิติรัตน์ ศิริบวรรัตนกุล เป็นผู้สอน

โดยทำการศึกษาเพื่อหาโมเดลต่างๆของ Convolutional Neural Network (CNN) ที่สามารถทำการแยกรูปภาพขยะแต่ละประเภทได้ และเปรียบเทียบประสิทธิภาพการทำงานระหว่างโมเดลแต่ละประเภท

# Data
## Data source
ข้อมูลเป็นรูปภาพขยะ โดยทำการรวบรวมจาก search engine, ภาพจริง, dataset จาก kaggle ()
โดยเราจะทำการแยกขยะออกเป็น 4 ประเภท ตามสถาบัรพลาสติก กระทรวงอุตสาหกรรม ดังนี้

![MicrosoftTeams-image (9)](https://user-images.githubusercontent.com/85028821/195612748-2e4ba3eb-ef39-4c8d-b88a-53fb236c00bf.png)
Figure 1 (CP for Sustainability, 2020)

ดังนั้น ทางกลุ่มเราจึงเลือกหยิบกลุ่มตัวอย่างในขยะแต่ละประเภทมาดังนี้
- ประเภทที่1 : ขยะอินทรีย์ มี 3 กลุ่ม คือ เศษอาหารและเนื้อสัตว์ เศษผักและผลไม้ และเศษใบไม้
- ประเภทที่2 : ขยะรีไซเคิล มี 3 กลุ่ม คือ กระดาษ พลาสติก และอัลลูมิเนียม
- ประเภทที่3 : ขยะทั่วไป มี 4 กลุ่ม คือ บรรจุภัณฑ์ ทิชชู่ โฟม และหลอดดูดน้ำ
- ประเภทที่4 : ขยะอันตราย มี 5 กลุ่ม คือ ถ่านไฟฉาย ยาหมดอายุ กระป๋องเสปรย์ หลอดฟลูออเรสเซนต์ และหน้ากากอนามัยใช้แล้ว

โดยแต่ละประเภทขยะมีจำนวนรูปประมาณประเภทละ 200 รูป

## Cleansing data
กลุ่มเราทำการปรับรูปภาพ ดังนี้
- เปลี่ยนประเภทไฟล์เป็น .jpg 
- ปรับขนาดรูปเป็น 512*512 pixel

creating tools : https://www.iloveimg.com/

## Data preprocessing
### Import data + split data + scaling data
ทำการ import image dataset จาก google drive และ ทำ preprocessing input ก่อนนำไปใช้กับ model

```
data_dir = '/content/drive/MyDrive/hw2_DADS7202_photo_4class'

np.random.seed(1234)
tf.random.set_seed(5678)

# Defining data generator withour Data Augmentation
data_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale = 1/255., validation_split = 0.3)

train_data = data_gen.flow_from_directory(data_dir, 
                                          target_size = (224, 224), 
                                          batch_size = 700,
                                          subset = 'training',
                                          class_mode = 'binary')
test_data = data_gen.flow_from_directory(data_dir, 
                                        target_size = (224, 224), 
                                        batch_size = 300,
                                        subset = 'testing',
                                        class_mode = 'binary')
```

- ทำการ split data เป็น train และ test สัดส่วน 70:30
- rescale รูปภาพเป็น 224*224 pixel
- batch size = 700 ของ train data และ 300 ของ test data

## EDA
### Check data type and shape 
```
print( f"x_train: type={type(x_train)} , dtype={x_train.dtype} , shape={x_train.shape} , min={x_train.min(axis=None)} , max={x_train.max(axis=None)}" )
print( f"x_test: type={type(x_test)} , dtype={x_test.dtype} , shape={x_test.shape} , min={x_test.min(axis=None)} , max={x_test.max(axis=None)}" )
```
ใส่รูป

### Check data distribution
class1 เป็น ขยะอินทรีย์ แทนด้วยค่า 0.0
class2 เป็น ขยะรีไซเคิล แทนด้วยค่า 1.0
class3 เป็น ขยะทั่วไป แทนด้วยค่า 2.0
class4 เป็น ขยะอันตราย แทนด้วยค่า 3.0

```
df_train = pd.DataFrame(y_train, columns = ['class'])
df_test = pd.DataFrame(y_test, columns = ['class'])

df_train_count = pd.DataFrame(df_train.groupby(['class'])['class'].count())

df_train_count.plot.bar()
```
![image](https://user-images.githubusercontent.com/85028821/195630526-c940029c-ee1c-4782-8624-44227d222843.png)

จากกราฟแสดงให้เห็นปริมาณข้อมูลของแต่ละclass ของ train dataset ว่ามีการกระจายตัวที่ใกล้เคียงกัน

```
df_test_count = pd.DataFrame(df_test.groupby(['class'])['class'].count())
df_test_count

df_test_count.plot.bar()
```
![image](https://user-images.githubusercontent.com/85028821/195630585-02cdf80d-7dae-4b95-90f7-c1cc4c2ccb87.png)

จากกราฟแสดงให้เห็นปริมาณข้อมูลของแต่ละclass ของ test dataset ว่ามีการกระจายตัวที่ใกล้เคียงกัน เช่นกัน

```
https://medium.com/geekculture/eda-for-image-classification-dcada9f2567a
```

### Visualize the images in x_train
```
plt.figure(figsize=(10,5))
for i in range(0,10):
    plt.title( f"{class_names[ int(y_train[i]) ]} [true = {int(y_train[i])} ]") 
    plt.imshow( x_train[i] )  
    plt.tight_layout()       
    plt.show()
```
ใส่รูป

# Model
## Use original model (Imagenet VGG-16)
### Prepare for transfer learning
ทำการโหลด Imagenet VGG-16 model มาใช้ โดยไม่เอาในส่วนของ classifier มา
```
img_w,img_h = 224,224
vgg_extractor = tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_w, img_h, 3))
vgg_extractor.summary()
```
Model: "vgg16"

 |Layer (type)            |    Output Shape          |    Param |   
|--------------------------|--------------------------|-------------|
 |input_3 (InputLayer)    |    [(None, 224, 224, 3)]  |   0     |                                                                     
 |block1_conv1 (Conv2D)   |    (None, 224, 224, 64)   |   1792   |                                                                    
 |block1_conv2 (Conv2D)   |    (None, 224, 224, 64)    |  36928   |                                                                 
 |block1_pool (MaxPooling2D) | (None, 112, 112, 64)    |  0        |                                                                  
 |block2_conv1 (Conv2D)   |    (None, 112, 112, 128)   |  73856     |                                                                 
 |block2_conv2 (Conv2D)    |   (None, 112, 112, 128)   |  147584    |                                                                 
 |block2_pool (MaxPooling2D) | (None, 56, 56, 128)     |  0         |                                                                 
 |block3_conv1 (Conv2D)   |    (None, 56, 56, 256)    |   295168    |                                                                 
 |block3_conv2 (Conv2D)   |    (None, 56, 56, 256)    |   590080    |                                                                 
 |block3_conv3 (Conv2D)    |   (None, 56, 56, 256)     |  590080    |                                                                 
 |block3_pool (MaxPooling2D) | (None, 28, 28, 256)    |   0         |                                                                 
 |block4_conv1 (Conv2D)   |    (None, 28, 28, 512)    |   1180160   |                                                                 
 |block4_conv2 (Conv2D)    |   (None, 28, 28, 512)     |  2359808   |                                                                 
 |block4_conv3 (Conv2D)    |   (None, 28, 28, 512)      | 2359808   |                                                                
 |block4_pool (MaxPooling2D) | (None, 14, 14, 512)     |  0         |                                                               
 |block5_conv1 (Conv2D)    |   (None, 14, 14, 512)    |   2359808   |                                                                
 |block5_conv2 (Conv2D)   |    (None, 14, 14, 512)     |  2359808   |                                                                
 |block5_conv3 (Conv2D)    |   (None, 14, 14, 512)      | 2359808   |                                                                
 |block5_pool (MaxPooling2D) | (None, 7, 7, 512)        | 0         |                                                              

- Total params: 14,714,688
- Trainable params: 14,714,688
- Non-trainable params: 0

ทำการ freeze layer ทั้งหมดใน feature extractor
```
vgg_extractor.trainable = False

for i,layer in enumerate(vgg_extractor.layers):  
    print( f"Layer {i}: name = {layer.name} , trainable = {layer.trainable}" )
```
Layer 0: name = input_15 , trainable = False
Layer 1: name = block1_conv1 , trainable = False
Layer 2: name = block1_conv2 , trainable = False
Layer 3: name = block1_pool , trainable = False
Layer 4: name = block2_conv1 , trainable = False
Layer 5: name = block2_conv2 , trainable = False
Layer 6: name = block2_pool , trainable = False
Layer 7: name = block3_conv1 , trainable = False
Layer 8: name = block3_conv2 , trainable = False
Layer 9: name = block3_conv3 , trainable = False
Layer 10: name = block3_pool , trainable = False
Layer 11: name = block4_conv1 , trainable = False
Layer 12: name = block4_conv2 , trainable = False
Layer 13: name = block4_conv3 , trainable = False
Layer 14: name = block4_pool , trainable = False
Layer 15: name = block5_conv1 , trainable = False
Layer 16: name = block5_conv2 , trainable = False
Layer 17: name = block5_conv3 , trainable = False
Layer 18: name = block5_pool , trainable = False

ทำการเพิ่มส่วนของ classifier ตาม model ของ VGG16 ใน Keras โดย layer สุดท้ายจะมีการจำแนกข้อมูลเป็น 4 class เนื่องจาก เราต้องการทำนายรูปภาพขยะออกเป็น 4 ประเภท

```
x = vgg_extractor.output

# Add our custom layer(s) to the end of the existing model 
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation="relu")(x)
x = tf.keras.layers.Dense(4096, activation="relu")(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)
model.summary()
```

Model: "model_18"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 input_15 (InputLayer)       [(None, 224, 224, 3)]     0         
                                                                 
 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
 flatten_7 (Flatten)         (None, 25088)             0         
                                                                 
 dense_25 (Dense)            (None, 4096)              102764544 
                                                                 
 dense_26 (Dense)            (None, 4096)              16781312  
                                                                 
 dense_27 (Dense)            (None, 4)                 16388     
                                                                 
=================================================================
- Total params: 134,276,932
- Trainable params: 119,562,244
- Non-trainable params: 14,714,688

### Train the model with transfer learning and set seed

ทำการเอาข้อมูลไปเข้า preprocessing ก่อนนำไปใช้ใน model
```
np.random.seed(1234)
tf.random.set_seed(5678)

# Defining data generator withour Data Augmentation
data_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale = 1/255., validation_split = 0.3)

train_data = data_gen.flow_from_directory(data_dir, 
                                          target_size = (224, 224), 
                                          batch_size = 700,
                                          subset = 'training',
                                          class_mode = 'binary')
test_data = data_gen.flow_from_directory(data_dir, 
                                        target_size = (224, 224), 
                                        batch_size = 300,
                                        subset = 'validation',
                                        class_mode = 'binary')
x_train, y_train = train_data.next()
x_test, y_test = test_data.next()
```

กำหนด Arguments ต่างๆของ model 
```
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )
```
- ค่า loss ใช้ sparse_categorical_crossentropy
- optimizer เป็น Adam
- metrics เป็น accuracy

ทำการ run model ด้วย x_train และ y_train และมีการกำหนดให้เลือก weight ที่ให้ค่า accuracy มากสุดไปใช้ใน model สุดท้าย โดยใช้ callbacks
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=5, epochs=10, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

![image](https://user-images.githubusercontent.com/85028821/195815557-b07e583a-1857-42c8-a88f-0f7768d12907.png)

กราฟ accuracy และ กราฟ loss

```
# Summarize history for accuracy
plt.figure(figsize=(15,5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Train accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()
plt.show()

# Summarize history for loss
plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.grid()
plt.show()
```

![image](https://user-images.githubusercontent.com/85028821/195815934-5cd5277c-3474-4e4d-a75f-3a452db53365.png)

![image](https://user-images.githubusercontent.com/85028821/195817457-32f46fab-307a-47ea-a65f-15be86a4d69a.png)


### Evaluate on test set 
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```


### Evaluate on test set without seed
ทำการเอา set seed ในการ train ออก แล้วทำการสร้าง model และ run train กับ test ใหม่ เพื่อหาค่าเฉลี่ยของ accuracy บน test set โดยทำทั้งหมด 3 รอบ
```
# create model
img_w,img_h = 224,224
vgg_extractor = tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_w, img_h, 3))
vgg_extractor.trainable = False
x = vgg_extractor.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation="relu")(x)
x = tf.keras.layers.Dense(4096, activation="relu")(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)

#train model without seed
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )

start_time = datetime.now()

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=5, epochs=10, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

#Evaluate on test set without seed
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
ผลลัพท์ accuracy บน test set 3 รอบคือ
1.
2.
3.

## Use .... model

Reference
- CP for Sustainability, 2020, accessed 13 Oct 2022, <https://www.sustainablelife.co/news/detail/74>
