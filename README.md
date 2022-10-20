# Introduction

จุดประสงค์การศึกษาครั้งนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep learning คณะสถิติประยุกต์ สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล(DADS) สถาบันบัณฑิตพัฒนบริหารศาสตร์ โดยมี ผศ.ดร.ฐิติรัตน์ ศิริบวรรัตนกุล เป็นผู้สอน

โดยทำการศึกษาเพื่อหาโมเดลต่างๆของ Convolutional Neural Network (CNN) ที่สามารถทำการแยกรูปภาพขยะแต่ละประเภทได้ และเปรียบเทียบประสิทธิภาพการทำงานระหว่างโมเดลแต่ละประเภท

# Data
## Data source
ข้อมูลเป็นรูปภาพขยะ โดยทำการรวบรวมจาก search engine, ภาพจริง, dataset จาก kaggle (https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

โดยเราจะทำการแยกขยะออกเป็น 4 ประเภท ตามหลักเกณฑ์ของสถาบันพลาสติก กระทรวงอุตสาหกรรม ดังนี้

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
data_gen = ImageDataGenerator(rescale = 1/255., validation_split = 0.3)

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
![image](https://user-images.githubusercontent.com/85028821/196121107-027efaba-560d-4904-afee-94c1bb4e94cb.png)

### Check data distribution
- class1 เป็น ขยะอินทรีย์ แทนด้วยค่า 0.0
- class2 เป็น ขยะรีไซเคิล แทนด้วยค่า 1.0
- class3 เป็น ขยะทั่วไป แทนด้วยค่า 2.0
- class4 เป็น ขยะอันตราย แทนด้วยค่า 3.0

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

### Visualize the images in x_train
```
plt.figure(figsize=(10,5))
for i in range(0,10):
    plt.title( f"{class_names[ int(y_train[i]) ]} [true = {int(y_train[i])} ]") 
    plt.imshow( x_train[i] )  
    plt.tight_layout()       
    plt.show()
```
![image](https://user-images.githubusercontent.com/85028821/196135569-abdf2d78-f8a6-42e5-9e6c-f2eb927c5d40.png)

# 1. VGG-16
## 1.1 Original Pre-trained model (VGG-16)
### Create the base model from the pre-trained convnets
ทำการโหลด Imagenet VGG-16 model มาใช้ โดยเอาในส่วนของ classifier มาด้วย และลบ layer ที่แบ่งข้อมูลออกเป็น 1000 class
```
vgg_extractor = tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=True)

# delete last layer
from keras.models import Model
vgg_extractor= Model(inputs=vgg_extractor.input, outputs=vgg_extractor.layers[-2].output)
```
### Freeze the convolutional base
ทำการ freeze layer ทั้งหมดใน feature extractor
```
vgg_extractor.trainable = False
```

### Add a classification head
ทำการเพิ่มส่วนของ classifier ตาม model ของ VGG16 ใน Keras โดย layer สุดท้ายจะมีการจำแนกข้อมูลเป็น 4 class เนื่องจาก เราต้องการทำนายรูปภาพขยะออกเป็น 4 ประเภท

```
x = vgg_extractor.output

# Add our custom layer(s) to the end of the existing model 

new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)
```

Model flow

See in : https://user-images.githubusercontent.com/85028821/196149170-41bc46ce-3899-48ab-a2a1-2de71ea1c408.png


### Preprocessing input
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
### Compile the model
ทำการ compile กำหนด Arguments ต่างๆของ model 
```
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )
```
- ค่า loss ใช้ sparse_categorical_crossentropy
- optimizer เป็น Adam
- metrics เป็น accuracy

### Train the model
ทำการ run model ด้วย x_train และ y_train และมีการกำหนดให้เลือก weight ที่ให้ค่า accuracy มากสุดไปใช้ใน model สุดท้าย โดยใช้ callbacks
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=10, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

![image](https://user-images.githubusercontent.com/85028821/195815557-b07e583a-1857-42c8-a88f-0f7768d12907.png)

จะเห็นว่าในการ train ครั้งนี้ค่าที่ดีที่สุดของ accuracy อยู่ที่ 0.9227 และของ validation accuracy อยู่ที่ 0.8201 อยู่ใน epoch ที่ 3 โดยเราจะใช้โมเดลใน epoch อันนี้ ในการไปใช้กับ test set ต่อไป  

### Learning curves
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
![image](https://user-images.githubusercontent.com/85028821/196142277-c9d976e4-ca05-4d28-9276-c8b36a39be79.png)

ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่ 0.6208 

### Evaluate on test set without seed
ทำการเอา set seed ในการ train ออก แล้วทำการสร้าง model และ run train กับ test ใหม่ เพื่อหาค่าเฉลี่ยของ accuracy บน test set โดยทำทั้งหมด 3 รอบ
```
# create model
vgg_extractor = tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=True)
vgg_extractor= Model(inputs=vgg_extractor.input, outputs=vgg_extractor.layers[-2].output)
vgg_extractor.trainable = False

# add classifier
x = vgg_extractor.output
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)

#train model without seed
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)
history = model.fit( x_train , y_train, batch_size=10, epochs=10, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

#Evaluate on test set without seed
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
ผลลัพท์ accuracy บน test set 3 รอบคือ
- ครั้งที่ 1 มี accuracy = 0.6356
- ครั้งที่ 2 มี accuracy = 0.6468
- ครั้งที่ 3 มี accuracy = 0.5873

ค่าเฉลี่ย accuracy 3 รอบ ของ test set = 0.6232 

## 1.2 Tuning model (VGG-16)
### Create feature extractor
```
img_w,img_h = 224,224
vgg_extractor = tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_w, img_h, 3))
vgg_extractor.trainable = False
```

### Un-freeze the top layers of the model
ทำการ un-freeze 2 layer สุดท้ายของ feature extractor
```
vgg_extractor.layers[-2].trainable = True
vgg_extractor.layers[-1].trainable = True
```

### Add a classification head
ทำการเพิ่มในส่วนของ classifier ต่อท้ายกับส่วนของ feature extractor
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

Model flow

See in : https://user-images.githubusercontent.com/85028821/196160471-87944299-63d6-4516-8128-7c38e8c4a2a0.png

### Compile the model
ทำการ compile กำหนด Arguments ต่างๆของ model 
```
alpha = 0.001
model.compile( loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adamax(learning_rate = alpha) , metrics=["acc"] )
```
- ค่า loss ใช้ sparse_categorical_crossentropy
- optimizer เป็น Adamax กำหนดค่า learning rate เป็น 0.01
- metrics เป็น accuracy

### Train the model
ทำการ run model ด้วย x_train และ y_train และมีการกำหนดให้เลือก weight ที่ให้ค่า accuracy มากสุดไปใช้ใน model สุดท้าย โดยใช้ callbacks
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=10, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
https://user-images.githubusercontent.com/85028821/196152611-4cabb8af-7476-47eb-a199-44c7f4af32fe.png

จะเห็นว่าในการ train ครั้งนี้ค่าที่ดีที่สุดของ accuracy อยู่ที่ 0.9909 และของ validation accuracy อยู่ที่ 0.8360 อยู่ใน epoch ที่ 4 โดยเราจะใช้โมเดลใน epoch อันนี้ ในการไปใช้กับ test set ต่อไป  

### Learning curves
กราฟ accuracy และ กราฟ loss

![image](https://user-images.githubusercontent.com/85028821/196158352-1b3acc69-ad29-463e-bc35-dc478a985d5f.png)
![image](https://user-images.githubusercontent.com/85028821/196158403-c6cfdf60-0eb9-4a30-be3f-1b5b20412b4e.png)

### Evaluate on test set
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
![image](https://user-images.githubusercontent.com/85028821/196158476-bb66cda2-8e8f-45f1-95a9-693f31055b2c.png)

ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่ 0.6914 

### Evaluate on test set without seed
ทำการเอา set seed ในการ train ออก แล้วทำการสร้าง model และ run train กับ test ใหม่ เพื่อหาค่าเฉลี่ยของ accuracy บน test set โดยทำทั้งหมด 3 รอบ
```
# create model
img_w,img_h = 224,224 
vgg_extractor = tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_w, img_h, 3))
vgg_extractor.trainable = False
vgg_extractor.layers[-2].trainable = True
vgg_extractor.layers[-1].trainable = True
x = vgg_extractor.output

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation="relu")(x)
x = tf.keras.layers.Dense(4096, activation="relu")(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)

#train model without seed
alpha = 0.001
model.compile( loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adamax(learning_rate = alpha) , metrics=["acc"] )

start_time = datetime.now()
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=10, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
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
- ครั้งที่ 1 มี accuracy = 0.7286
- ครั้งที่ 2 มี accuracy = 0.7323
- ครั้งที่ 3 มี accuracy = 0.6877

ค่าเฉลี่ย accuracy 3 รอบ ของ test set = 0.7162


# 2. RESNET50
## 2.1 Original Pre-trained model (RESNET50)
### Create the base model from the pre-trained convnets
ทำการโหลด Imagenet resnet50 model มาใช้ โดยเอาในส่วนของ classifier มาด้วย และลบ layer ที่แบ่งข้อมูลออกเป็น 1000 class
```
resnet50_extractor = tf.keras.applications.resnet50.ResNet50(include_top=True,weights='imagenet')
# delete last layer
from keras.models import Model
resnet50_extractor= Model(inputs=resnet50_extractor.input, outputs=resnet50_extractor.layers[-2].output)

```
### Freeze the convolutional base
ทำการ freeze layer ทั้งหมดใน feature extractor
```
resnet50_extractor.trainable = False
```

### Add a classification head
ทำการเพิ่มส่วนของ classifier ตาม model ของ resnet50 ใน Keras โดย layer สุดท้ายจะมีการจำแนกข้อมูลเป็น 4 class เนื่องจาก เราต้องการทำนายรูปภาพขยะออกเป็น 4 ประเภท

```
x = resnet50_extractor.output

# Add our custom layer(s) to the end of the existing model 

new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs=resnet50_extractor.inputs, outputs=new_outputs)

```

Model flow

See in : https://user-images.githubusercontent.com/80901294/196710788-333cefc7-0518-47ae-9346-24b3628965a4.png


### Preprocessing input
ทำการเอาข้อมูลไปเข้า preprocessing ก่อนนำไปใช้ใน model
```
np.random.seed(1234)
tf.random.set_seed(5678)

# Defining data generator withour Data Augmentation
data_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input, rescale = 1/255., validation_split = 0.3)

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

```
### Compile the model
ทำการ compile กำหนด Arguments ต่างๆของ model 
```
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )
```
- ค่า loss ใช้ sparse_categorical_crossentropy
- optimizer เป็น Adam
- metrics เป็น accuracy

### Train the model
ทำการ run model ด้วย x_train และ y_train และมีการกำหนดให้เลือก weight ที่ให้ค่า accuracy มากสุดไปใช้ใน model สุดท้าย โดยใช้ callbacks
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
<img width="662" alt="trainor" src="https://user-images.githubusercontent.com/80901294/196683018-b529e871-946b-4116-9b47-1309361aab26.png">


จะเห็นว่าในการ train ครั้งนี้ค่าที่ดีที่สุดของ accuracy อยู่ที่ 0.5915 และของ validation accuracy อยู่ที่ 0.54098อยู่ใน epoch ที่ 26 โดยเราจะใช้โมเดลใน epoch อันนี้ ในการไปใช้กับ test set ต่อไป  

### Learning curves
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
![ผลtrain](https://user-images.githubusercontent.com/80901294/196682961-a2ce57e5-8c08-465f-97aa-24b622474de6.png)

![Unknown](https://user-images.githubusercontent.com/80901294/196750662-ae737336-d87a-47ec-b772-a7c06a4d7fc6.png)


### Evaluate on test set
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
<img width="662" alt="test9or" src="https://user-images.githubusercontent.com/80901294/196683078-b374ec42-3d51-4143-a0e7-70fc3931b774.png">


ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่ 0.46899 

### Evaluate on test set without seed
ทำการเอา set seed ในการ train ออก แล้วทำการสร้าง model และ run train กับ test ใหม่ เพื่อหาค่าเฉลี่ยของ accuracy บน test set โดยทำทั้งหมด 3 รอบ
```
# create model
resnet50_extractor = tf.keras.applications.resnet50.ResNet50(weights = "imagenet", include_top=True)
resnet50_extractor= Model(inputs=resnet50_extractor.input, outputs=resnet50_extractor.layers[-2].output)
resnet50_extractor.trainable = False

# add classifier
x = resnet50_extractor.output
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
model = tf.keras.models.Model(inputs=resnet50_extractor.inputs, outputs=new_outputs)

#train model without seed
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)
history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

#Evaluate on test set without seed
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

```
ผลลัพท์ accuracy บน test set 3 รอบคือ
- ครั้งที่ 1 มี accuracy = 0.3643410801887512
- ครั้งที่ 2 มี accuracy = 0.3759689927101135
- ครั้งที่ 3 มี accuracy = 0.44186046719551086

ค่าเฉลี่ย accuracy 3 รอบ ของ test set  = 0.394

## 2.2 Tuning model (RESNET50)
### Create feature extractor
```
img_w,img_h = 224,224
resnet50_extractor = tf.keras.applications.resnet50.ResNet50(include_top=True,weights='imagenet', input_shape = (img_w, img_h, 3))
resnet50_extractor.trainable = False
```

### Un-freeze the top layers of the model
```
vgg_extractor.layers[-2].trainable = True
vgg_extractor.layers[-1].trainable = True
```

### Add a classification head
```
x = resnet50_extractor.layers[-2].output

# Add our custom layer(s) to the end of the existing model 
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(8192, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs=resnet50_extractor.inputs, outputs=new_outputs)
model.summary()
```

Model flow

See in :https://user-images.githubusercontent.com/80901294/196683845-4eafa845-3d58-4167-ab87-dba185f255e1.png

### Compile the model
ทำการ compile กำหนด Arguments ต่างๆของ model 
```
alpha = 0.001
model.compile( loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adamax(learning_rate = alpha) , metrics=["acc"] )
```
- ค่า loss ใช้ sparse_categorical_crossentropy
- optimizer เป็น Adamax กำหนดค่า learning rate เป็น 0.01
- metrics เป็น accuracy

### Train the model
ทำการ run model ด้วย x_train และ y_train และมีการกำหนดให้เลือก weight ที่ให้ค่า accuracy มากสุดไปใช้ใน model สุดท้าย โดยใช้ callbacks
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
<img width="662" alt="trainor" src="https://user-images.githubusercontent.com/80901294/196683416-84083b16-9c0e-40ad-aa2a-f3046879b242.png">

จะเห็นว่าในการ train ครั้งนี้ค่าที่ดีที่สุดของ accuracy อยู่ที่ 0.5117 และของ validation accuracy อยู่ที่ 0.54645 อยู่ใน epoch ที่ 23 โดยเราจะใช้โมเดลใน epoch อันนี้ ในการไปใช้กับ test set ต่อไป  

### Learning curves
กราฟ accuracy และ กราฟ loss
![ผลtrain](https://user-images.githubusercontent.com/80901294/196683489-f0879f5a-549e-403e-ae81-8a1cdd2bc20a.png)

![Unknown-2](https://user-images.githubusercontent.com/80901294/196751099-edc7c9db-ced5-4a3f-ac70-c7d8ca48f2b3.png)


### Evaluate on test set!
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
<img width="662" alt="Screen Shot 2565-10-19 at 18 43 38" src="https://user-images.githubusercontent.com/80901294/196683560-c191c10c-5f44-4d66-849f-6fa6d7a945ce.png">


ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่ 0.46124 

### Evaluate on test set without seed
ทำการเอา set seed ในการ train ออก แล้วทำการสร้าง model และ run train กับ test ใหม่ เพื่อหาค่าเฉลี่ยของ accuracy บน test set โดยทำทั้งหมด 3 รอบ

```
# create model
img_w,img_h = 224,224 
resnet50_extractor = tf.keras.applications.resnet50.ResNet50(weights = "imagenet", include_top=True, input_shape = (img_w, img_h, 3))

x = resnet50_extractor.layers[-2].output

# Add our custom layer(s) to the end of the existing model 
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(8192, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

model = tf.keras.models.Model(inputs=resnet50_extractor.inputs, outputs=new_outputs)

#train model without seed
alpha = 0.001
model.compile( loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adamax(learning_rate = alpha) , metrics=["acc"] )

start_time = datetime.now()
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
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
- ครั้งที่ 1 มี accuracy = 0.7209302186965942
- ครั้งที่ 2 มี accuracy = 0.6976743936538696
- ครั้งที่ 3 มี accuracy = 0.7093023061752319

ค่าเฉลี่ย accuracy 3 รอบ ของ test set  = 0.7093


# 3. MobileNet
## 3.1 Original Pre-trained model (MobileNet)
### Create the base model from the pre-trained convnets
ทำการโหลด Imagenet MobileNet model มาใช้ โดยเอาในส่วนของ classifier มาด้วย และลบ layer ที่แบ่งข้อมูลออกเป็น 1000 class
```
vgg_extractor = tf.keras.applications.mobilenet.MobileNet(weights = "imagenet", include_top=False)

# delete last layer
from keras.models import Model
vgg_extractor= Model(inputs=vgg_extractor.input, outputs=vgg_extractor.layers[-2].output)
vgg_extractor.summary()
```

### Freeze the convolutional base
```
vgg_extractor.trainable = False

for i,layer in enumerate(vgg_extractor.layers):  
    print( f"Layer {i}: name = {layer.name} , trainable = {layer.trainable}" )```
```

### Add a classification head
```
x = vgg_extractor.output

# Add our custom layer(s) to the end of the existing model 
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)
model.summary()
```

Model flow

See in : https://user-images.githubusercontent.com/97573140/196418863-035da0d4-ed5b-49b5-b1bc-28a323c43603.png


### Preprocessing input
```
np.random.seed(1234)
tf.random.set_seed(5678)

# Defining data generator withour Data Augmentation
data_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input, rescale = 1/255., validation_split = 0.3)

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
### Compile the model
```
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )
```

### Train the model
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

![ori-epoch](https://user-images.githubusercontent.com/97573140/196720382-03128ee4-b573-4cd6-bc59-6f199efb6eca.png)


### Learning curves
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
![ori-graph1](https://user-images.githubusercontent.com/97573140/196716416-4eae4510-b50c-461b-87ac-2e8364b9f76c.png)
![ori-graph2](https://user-images.githubusercontent.com/97573140/196716647-395eaa4e-b308-426c-8a79-acbf43598fb6.png)


### Evaluate on test set
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

![ori-acc](https://user-images.githubusercontent.com/97573140/196717104-17a309f0-ffab-4ce8-83c0-2575cc597113.png)


### Evaluate on test set without seed
```
# create model
vgg_extractor = tf.keras.applications.mobilenet.MobileNet(weights = "imagenet", include_top=True)
vgg_extractor= Model(inputs=vgg_extractor.input, outputs=vgg_extractor.layers[-2].output)
vgg_extractor.trainable = False

# add classifier
x = vgg_extractor.output
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)

#train model without seed
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)
history = model.fit( x_train , y_train, batch_size=10, epochs=50, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

#Evaluate on test set without seed
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
ผลลัพท์ accuracy บน test set 3 รอบคือ

- ครั้งที่ 1 มี accuracy = 0.3233
- ครั้งที่ 2 มี accuracy = 0.3233
- ครั้งที่ 3 มี accuracy = 0.3300

ค่าเฉลี่ย accuracy 3 รอบ ของ test set = 0.3255

## 3.2 Tuning model (MobileNet)
### Create feature extractor
```
img_w,img_h = 224,224
vgg_extractor = tf.keras.applications.mobilenet.MobileNet(weights = "imagenet", include_top=True, input_shape = (img_w, img_h, 3))

vgg_extractor.summary()
```
### Add a classification head
```
x = vgg_extractor.layers[-5].output

# Add our custom layer(s) to the end of the existing model 
#x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation="relu")(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)
model.summary()
```

Model flow

See in : https://user-images.githubusercontent.com/97573140/196419603-11948fa6-f732-455d-9bac-7c456f63b094.png


### Compile the model
```
alpha = 0.001
model.compile( loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adamax(learning_rate = alpha) , metrics=["acc"] )
```

### Train the model
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

![tune-train-epoch](https://user-images.githubusercontent.com/97573140/196420774-cc0dd6f4-0d47-48fb-9d9d-70ac202e0fbe.PNG)

### Learning curves
```
#Summarize history for accuracy

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
กราฟ accuracy และ กราฟ loss
![tune-graph1](https://user-images.githubusercontent.com/97573140/196419854-c4fc0252-dac5-443c-a5b3-013fa2ffc699.png)
![tune-graph2](https://user-images.githubusercontent.com/97573140/196419869-55b1dfce-1b61-4f2f-9308-ffb0a1f40963.png)


### Evaluate on test set
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

![tune-acc](https://user-images.githubusercontent.com/97573140/196420395-c65c3971-966e-49cc-8609-4edcafe66ff4.PNG)

ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่ 0.56333

### Evaluate on test set without seed
```
# create model
img_w,img_h = 224,224 
vgg_extractor = tf.keras.applications.mobilenet.MobileNet(weights = "imagenet", include_top=True, input_shape = (img_w, img_h, 3))
#vgg_extractor.trainable = False
#vgg_extractor.layers[-2].trainable = True
#vgg_extractor.layers[-1].trainable = True

x = vgg_extractor.layers[-5].output

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation="relu")(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)

#train model without seed
alpha = 0.001
model.compile( loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adamax(learning_rate = alpha) , metrics=["acc"] )

start_time = datetime.now()
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
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
- ครั้งที่ 1 มี accuracy = 0.6133
- ครั้งที่ 2 มี accuracy = 0.5400
- ครั้งที่ 3 มี accuracy = 0.5733

ค่าเฉลี่ย accuracy 3 รอบ ของ test set  = 0.5755


# EfficientNet V2
## 4.1 Original Pre-trained model (EfficientNet V2 B1)
### Create the base model from the pre-trained convnets
ทำการโหลด Imagenet EfficientNet V2 model มาใช้ โดยเอาในส่วนของ classifier มาด้วย และลบ layer ที่แบ่งข้อมูลออกเป็น 1000 class
```
effnet_v2 = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape= None ,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    include_preprocessing=True)
```

```
# delete last layer
from keras.models import Model
effnet_v2= Model(inputs=effnet_v2.input, outputs=effnet_v2.layers[-2].output)
```
### Freeze the convolutional base
ทำการ freeze layer ทั้งหมดใน feature extractor
```
effnet_v2.trainable = False
```

### Add a classification head
ทำการเพิ่มส่วนของ classifier ตาม model ของ EfficientNetV2B1 ใน Keras โดย layer สุดท้ายจะมีการจำแนกข้อมูลเป็น 4 class เนื่องจาก เราต้องการทำนายรูปภาพขยะออกเป็น 4 ประเภท
```
x = effnet_v2.output

# Add our custom layer(s) to the end of the existing model 
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs= effnet_v2.inputs, outputs=new_outputs)
```

Model flow

See in : https://user-images.githubusercontent.com/97610480/196891983-9c9a0a7e-d0a6-4dca-b3d7-3e257a576490.png


### Preprocessing input
มีการทำ Data Augmentation เพราะช่วยให้ data set มีความ wary ขึ้น
```
# Defining data generator withour Data Augmentation
np.random.seed(1234)
tf.random.set_seed(5678)

# Defining data generator withour Data Augmentation
data_gen = ImageDataGenerator(rescale = 1./255,
                              validation_split = 0.3,
                              rotation_range = 40, 
                              horizontal_flip = True, 
                              width_shift_range = 0.2, 
                              height_shift_range = 0.2)

train_data = data_gen.flow_from_directory(data_dir, 
                                          target_size = (240, 240), 
                                          batch_size = 700,
                                          subset = 'training',
                                          class_mode = 'binary')
test_data = data_gen.flow_from_directory(data_dir, 
                                        target_size = (240, 240), 
                                        batch_size = 300,
                                        subset = 'validation',
                                        class_mode = 'binary')
```
### Compile the model
ทำการ compile กำหนด Arguments ต่างๆของ model 
```
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )
```
- ค่า loss ใช้ sparse_categorical_crossentropy
- optimizer เป็น Adam
- metrics เป็น accuracy

### Train the model
ทำการ run model ด้วย x_train และ y_train และมีการกำหนดให้เลือก weight ที่ให้ค่า accuracy มากสุดไปใช้ใน model สุดท้าย โดยใช้ callbacks
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=10, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

![image](https://user-images.githubusercontent.com/97610480/196892343-915cec2a-6daf-4f55-8aa2-913061f68dc3.png)


ผลลัพท์ที่ออกมาไม่ค่อยดีนัก เพราะได้ validation accuracy ที่ดีสุดแค่ 0.3736

### Learning curves
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

![image](https://user-images.githubusercontent.com/97610480/196892428-f2903676-3a23-4a92-b699-598287710aa2.png)

![image](https://user-images.githubusercontent.com/97610480/196892482-212097eb-8a08-4a40-b8da-7ca43b087345.png)


### Evaluate on test set
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
![image](https://user-images.githubusercontent.com/97610480/196892637-977049d2-448c-43d9-979b-123d2eed3216.png)

ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่ 0.353

### Evaluate on test set without seed
ทำการเอา set seed ในการ train ออก แล้วทำการสร้าง model และ run train กับ test ใหม่ เพื่อหาค่าเฉลี่ยของ accuracy บน test set โดยทำทั้งหมด 3 รอบ
```
# create model
effnet_v2 = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(weights = "imagenet", include_top=True)
effnet_v2= Model(inputs=effnet_v2.input, outputs=effnet_v2.layers[-2].output)
effnet_v2.trainable = False

# add classifier
x = effnet_v2.output
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
model = tf.keras.models.Model(inputs=effnet_v2.inputs, outputs=new_outputs)

#train model without seed
model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"] )

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)
history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')
#Evaluate on test set without seed
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
ผลลัพท์ accuracy บน test set 3 รอบคือ
- ครั้งที่ 1 มี accuracy = 0.3381
- ครั้งที่ 2 มี accuracy = 0.3197
- ครั้งที่ 3 มี accuracy = 0.2973

ค่าเฉลี่ยของ accuracy ที่ได้คือประมาณ 0.3184

## 4.2 Tuning model (EfficientNetV2B1)
### Create feature extractor
```
img_w,img_h = 240,240
effnet_v2 = effnet_v2 = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape= (img_w, img_h, 3) ,
    pooling="max",
    classes=1000,
    classifier_activation='softmax',
    include_preprocessing=True
)
effnet_v2.trainable = True
```
จากการทดลอง ผลลัพท์ที่ดีที่สุดของ Data set ชุดนี้ ได้มาจากการ unfreeze ทุก layers 

### Add a classification head
ทำการเพิ่มในส่วนของ classifier ต่อท้ายกับส่วนของ feature extractor
```

# Add our custom layer(s) to the end of the existing model 
x = effnet_v2.output

# Add our custom layer(s) to the end of the existing model 
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(8192, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs=effnet_v2.inputs, outputs=new_outputs)
# Construct the main model 
model = tf.keras.models.Model(inputs=effnet_v2.inputs, outputs=new_outputs)
model.summary()
```

Model flow

See in : https://user-images.githubusercontent.com/97610480/196896696-ed66f168-eb77-4bb9-898c-115c2c0f026e.png


### Compile the model
ทำการ compile กำหนด Arguments ต่างๆของ model 
```
opt = tf.keras.optimizers.Adamax(learning_rate = 0.001)
model.compile( loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"])
```

- ค่า loss ใช้ sparse_categorical_crossentropy
- optimizer เป็น Adamax กำหนดค่า learning rate เป็น 0.001
- metrics เป็น accuracy

### Train the model
ทำการ run model ด้วย x_train และ y_train และมีการกำหนดให้เลือก weight ที่ให้ค่า accuracy มากสุดไปใช้ใน model สุดท้าย โดยใช้ callbacks
```
from datetime import datetime
start_time = datetime.now()

np.random.seed(1234)
tf.random.set_seed(5678)

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=10, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

![image](https://user-images.githubusercontent.com/97610480/196896847-4ab27b2c-3cfe-44a1-a3d5-73b92a3b16a6.png)

จากการ Fine-tuning ในครั้งนี้เราได้ accuracy สูงสุดที่ 0.8158 ซึ่งดีกว่า based model ค่อนข้างเยอะ

### Learning curves
กราฟ accuracy และ กราฟ loss

![image](https://user-images.githubusercontent.com/97610480/196897127-3f5d17c2-136a-47e0-8e3d-35a683ade065.png)
![image](https://user-images.githubusercontent.com/97610480/196897060-d249d71b-4857-4409-97cb-a31f376b57ea.png)


### Evaluate on test set
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
![image](https://user-images.githubusercontent.com/97610480/196897204-9cad4e82-7c89-4135-882d-94cde623a9a8.png)

ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่ 0.6914

### Evaluate on test set without seed
ทำการเอา set seed ในการ train ออก แล้วทำการสร้าง model และ run train กับ test ใหม่ เพื่อหาค่าเฉลี่ยของ accuracy บน test set โดยทำทั้งหมด 3 รอบ
```
# create model
img_w,img_h = 240,240
effnet_v2 = effnet_v2 = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape= (img_w, img_h, 3) ,
    pooling="max",
    classes=1000,
    classifier_activation='softmax',
    include_preprocessing=True
)
effnet_v2.trainable = True
x = effnet_v2.output
```
```
# Add our custom layer(s) to the end of the existing model 
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(8192, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
```
```
# Construct the main model 
model = tf.keras.models.Model(inputs=effnet_v2.inputs, outputs=new_outputs)
```
```
opt = tf.keras.optimizers.Adamax(learning_rate = 0.001)
model.compile( loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"])
```
```
from datetime import datetime
start_time = datetime.now()

from keras import callbacks

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", monitor = 'val_acc', verbose=1, save_best_only=True)

history = model.fit( x_train , y_train, batch_size=10, epochs=30, verbose=1, validation_split=0.3, callbacks=[checkpointer] )
model.load_weights('weights.hdf5')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```

ผลลัพท์ accuracy บน test set 3 รอบคือ
- ครั้งที่ 1 มี accuracy = 0.6988
- ครั้งที่ 2 มี accuracy = 0.6319
- ครั้งที่ 3 มี accuracy = 0.6505

ค่าเฉลี่ย accuracy 3 รอบ ของ test set = 0.0.6604


# Conclusion

จากการทำ CNN ทางกลุ่มเราเลือกโมเดลมา 4 แบบ ตามจำนวนสมาชิค ดังนี้

1.VGG-16

<img width="429" alt="Screen Shot 2565-10-19 at 22 32 25" src="https://user-images.githubusercontent.com/80901294/196737506-af51d89f-292e-4bba-bc4c-543e4b882ed4.png">

ทำการประเมิณโมเดล จาก accuracy หลังจากการทำการปรับmodelแล้ว โดยได้ค่าaccuracyที่ดีขึ้น ใน test set without seed เพิ่มขึ้นจาก  0.6232 เป็น 0.7162 หรือประมาณ 15% โดยวิธีการปรับโมเดลมีดังนี้ ทำการ unfreeze 2 layer สุดท้ายของ feature extractor และปรับ optimizer เป็น Adamax

2.RESNET50

<img width="429" alt="Screen Shot 2565-10-19 at 22 32 38" src="https://user-images.githubusercontent.com/80901294/196737577-f3e03126-cf99-448a-8702-a6e6549a81f1.png">

ทำการประเมิณโมเดล จาก accuracy หลังจากการทำการปรับmodelแล้ว โดยได้ค่าaccuracyที่ดีขึ้น ใน test set without seed เพิ่มขึ้นจาก  0.3940 เป็น 0.7093 หรือประมาณ 80% โดยวิธีการปรับโมเดลมีดังนี้ ไม่ได้ทำการ unfreeze feature extractor ทั้งหมด, ปรับ optimizer เป็น Adamax และเพิ่ม classifier layer ดังนี้ เพิ่ม 2 dense layer มี acivation เป็น ReLu และขั้นด้วย DropOut ทุก layer

3.MobileNet

<img width="429" alt="Screen Shot 2565-10-19 at 22 32 47" src="https://user-images.githubusercontent.com/80901294/196737673-5a24daa9-d615-42c2-a58d-74c64a4e670c.png">

ทำการประเมิณโมเดล จาก accuracy หลังจากการทำการปรับmodelแล้ว โดยได้ค่าaccuracyที่ดีขึ้น ใน test set without seed เพิ่มขึ้นจาก  0.3255 เป็น 0.5755 หรือประมาณ 77% โดยวิธีการปรับโมเดลมีดังนี้ ไม่ได้ทำการ unfreeze feature extractor ทั้งหมด, ปรับ optimizer เป็น Adamax และเพิ่ม classifier layer ดังนี้ เพิ่ม 2 dense layer มี acivation เป็น ReLu และ DropOut

4.EfficiantNetV2

<img width="504" alt="Screen Shot 2565-10-20 at 15 53 03" src="https://user-images.githubusercontent.com/80901294/196903292-d4c009c4-f3ee-4475-ba6d-a33753fecfb7.png">

ทำการประเมิณโมเดล จาก accuracy หลังจากการทำการปรับmodelแล้ว โดยได้ค่าaccuracyที่ดีขึ้น ใน test set without seed เพิ่มขึ้นจาก  0.3184 เป็น 0.6604 หรือประมาณ 107% 
โดยวิธีการปรับโมเดลมีดังนี้ ทำการ unfreeze feature extractor ทั้งหมด, ปรับ optimizer เป็น Adamax, มีการทำ Data Augmentation ตอน  Pre-Processing input 
และเพิ่ม classifier layer ดังนี้ เพิ่ม 2 dense layer มี acivation เป็น ReLu และขั้นด้วย DropOut ทุก layer

จากข้อมูลข้างต้น โมเดลที่ให้ค่า accuracy กับ รูปภาพของกลุ่มเรา โดยเรียงจากมากสุดไปน้อยสุด คือ VGG-16, Resnet50, EfficiantNetV2 และ  MobileNet และในทุกโมเดลที่ได้ทำการปรับแล้ว จะทำให้ค่า accuracy ของแต่ละโมเดลดีขึ้น ส่วนระยะเวลาในการ runโมเดล สังเกตได้ว่า VGG-16 ใช้ระยเวลาในการ run ที่ค่อนข้างนานกว่าโมเดลอื่น คาดว่าเนื่องจากว่าจำนวน parameter มีจำนวนมากกว่าโมเดลอื่นมาก


# Member 
- นส.ศิริวลัย   มณีสินธุ์    6410412011
- นายศิวกร    ศรีชัยพฤกษ์ 6410412012
- นายนนทพร  วงษ์เล็ก    6410412016
- นายธีรพล   แสงเมือง    6410412019

Reference
- CP for Sustainability, 2020, accessed 13 Oct 2022, <https://www.sustainablelife.co/news/detail/74>
- https://www.iloveimg.com (resizing tools)
- François Chollet, 2017, accessed 17 Oct 2022, <https://www.tensorflow.org/tutorials/images/transfer_learning#fine_tuning>
- deeplizard, (2020 Sept 28), Fine-Tuning MobileNet on Custom Data Set with TensorFlow's Keras API [video].youtube.https://www.youtube.com/watch?v=Zrt76AIbeh4
- Tammina, S. (2019). Transfer learning using vgg-16 with deep convolutional neural network for classifying images. International Journal of Scientific and Research Publications (IJSRP), 9(10), 143-150.
- Mukti, I. Z., & Biswas, D. (2019, December). Transfer learning based plant diseases detection using ResNet50. In 2019 4th International conference on electrical information and communication technology (EICT) (pp. 1-6). IEEE.
- Gavai, N. R., Jakhade, Y. A., Tribhuvan, S. A., & Bhattad, R. (2017, December). MobileNets for flower classification using TensorFlow. In 2017 international conference on big data, IoT and data science (BID) (pp. 154-158). IEEE.
