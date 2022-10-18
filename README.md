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

See in : https://user-images.githubusercontent.com/85028821/196149170-41bc46ce-3899-48ab-a2a1-2de71ea1c408.png)


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
1. ['loss', 'acc']: [1.6552209854125977, 0.6356877088546753]
Duration: 0:00:01.581440
2. ['loss', 'acc']: [2.407459020614624, 0.6468401551246643]
Duration: 0:00:01.448805
3. ['loss', 'acc']: [1.997439980506897, 0.5873606204986572]
Duration: 0:00:01.460710

ค่าเฉลี่ย accuracy 3 รอบ ของ test set = 0.6232 

## 1.2 Tuning model (VGG-16)
### Create feature extractor
```
img_w,img_h = 224,224
vgg_extractor = tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_w, img_h, 3))
vgg_extractor.trainable = False
```

### Un-freeze the top layers of the model
```
vgg_extractor.layers[-2].trainable = True
vgg_extractor.layers[-1].trainable = True
```

### Add a classification head
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
1.
2.
3.

# 2. Resnet50
## Original Pre-trained model (Resnet50)
## Tuning model (Resnet50)


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
ขาดรูปepoch
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
ขาดรูปกราฟ

### Evaluate on test set
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
ขาดรูปผลการรัน
ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่ 0.6208 

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
1. 
2. 
3. 
ค่าเฉลี่ย accuracy 3 รอบ ของ test set = 

## 1.2 Tuning model (MobileNet)
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
ขาดรูป

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
ขาดรูปepoch

### Learning curves
กราฟ accuracy และ กราฟ loss
ขาดรูป

### Evaluate on test set
```
# Evaluate the trained model on the test set
start_time = datetime.now()

results = model.evaluate(x_test, y_test, batch_size=32)
print( f"{model.metrics_names}: {results}" )

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
```
ขาดรูป
ค่า accuracy เมื่อทำการ evaluate บน test set ได้ค่าอยู่ที่  

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
1. 0.6133
2. 0.5400
3. 0.5733
ค่าเฉลี่ย accuracy 3 รอบ ของ test set  = 0.57553

# EfficiantNetV2
## Original Pre-trained model (MobileNet)
## Tuning model (MobileNet)

# Conclusion

ทำตาราง 8 row

เขียนสรุป

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
