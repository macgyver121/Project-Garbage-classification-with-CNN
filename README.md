# Introduction

จุดประสงค์การศึกษาครั้งนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep learning คณะสถิติประยุกต์ สาขาวิชาการวิเคราะห์ข้อมูลและวิทยาการข้อมูล(DADS) สถาบันบัณฑิตพัฒนบริหารศาสตร์ โดยมี ผศ.ดร.ฐิติรัตน์ ศิริบวรรัตนกุล เป็นผู้สอน

โดยทำการศึกษาเปรียบเทียบประสิทธิภาพการทำงานระหว่างโมเดลประเภทต่างๆของ  Convolutional Neural Network (CNN) โดยใช้ข้อมูลรูปภาพ เพื่อทำการแยกประเภทขยะ

# Data
## Data source
ข้อมูลเป็นรูปภาพขยะ โดยทำการรวบรวมจาก search engine, ภาพจริง, dataset จาก kaggle ()
โดยเราจะทำการแยกขยะออกเป็น 4 ประเภท ตามสถาบัรพลาสติก กระทรวงอุตสาหกรรม ดังนี้

![MicrosoftTeams-image (9)](https://user-images.githubusercontent.com/85028821/195612748-2e4ba3eb-ef39-4c8d-b88a-53fb236c00bf.png)

ดังนั้น ทางกลุ่มเราจึงเลือกหยิบกลุ่มตัวอย่างในขยะแต่ละประเภทมาดังนี้
- ประเภทที่1 : ขยะอินทรีย์ มี 3 กลุ่ม คือ เศษอาหารและเนื้อสัตว์ เศษผักและผลไม้ และเศษใบไม้
- ประเภทที่2 : ขยะรีไซเคิล มี 3 กลุ่ม คือ กระดาษ พลาสติก และอัลลูมิเนียม
- ประเภทที่3 : ขยะทั่วไป มี 5 กลุ่ม คือ บรรจุภัณฑ์ ถุงพลาสติก ทิชชู่ โฟม และหลอดดูดน้ำ
- ประเภทที่4 : ขยะอันตราย มี 5 กลุ่ม คือ ถ่านไฟฉาย ยาหมดอายุ กระป๋องเสปรย์ หลอดฟลูออเรสเซนต์ และหน้ากากอนามัยใช้แล้ว

โดยแต่ละประเภทขยะมีจำนวนรูปประมาณประเภทละ 200 รูป

## Cleansing data
กลุ่มเราทำการปรับรูปภาพ ดังนี้
- เปลี่ยนประเภทไฟล์เป็น .jpg 
- ปรับขนาดรูปเป็น 512*512 pixel

creating tools : https://www.iloveimg.com/

## Data preprocessing
### Import data + split data + scaling data
ทำการ import image dataset จาก google drive

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
.....

### Check data distribution
class1 เป็น ขยะอินทรีย์ อยู่ในตำแหน่ง index 0.0
class2 เป็น ขยะรีไซเคิล อยู่ในตำแหน่ง index 1.0
class3 เป็น ขยะทั่วไป อยู่ในตำแหน่ง index 2.0
class4 เป็น ขยะอันตราย อยู่ในตำแหน่ง index 3.0

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
## Use original model
### Prepare for transfer learning
ทำการโหลด Imagenet VGG-16 model มาใช้
```
vgg = tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=True)
vgg.summary()
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
 |flatten (Flatten)      |     (None, 25088)       |      0         |                                                                
 |fc1 (Dense)            |     (None, 4096)         |     102764544 |                                                                 
 |fc2 (Dense)            |     (None, 4096)          |    16781312  |                                                                 
| predictions (Dense)     |    (None, 1000)           |   4097000   |

- Total params: 138,357,544
- Trainable params: 138,357,544
- Non-trainable params: 0

ทำการลบ layer สุดท้ายออก 1 layer และดูว่าแต่ละ layer ถูก freeze หรือไม่
```
from keras.models import Model
vgg_extractor= Model(inputs=vgg.input, outputs=vgg.layers[-2].output)
vgg_extractor.summary()

for i,layer in enumerate(vgg_extractor.layers):  
    print( f"Layer {i}: name = {layer.name} , trainable = {layer.trainable}" )
```
Model: "model_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param   
=================================================================
 input_3 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
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
                                                                 
 flatten (Flatten)           (None, 25088)             0         
                                                                 
 fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
 fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
=================================================================
Total params: 134,260,544
Trainable params: 134,260,544
Non-trainable params: 0
_________________________________________________________________
Layer 0: name = input_3 , trainable = True
Layer 1: name = block1_conv1 , trainable = True
Layer 2: name = block1_conv2 , trainable = True
Layer 3: name = block1_pool , trainable = True
Layer 4: name = block2_conv1 , trainable = True
Layer 5: name = block2_conv2 , trainable = True
Layer 6: name = block2_pool , trainable = True
Layer 7: name = block3_conv1 , trainable = True
Layer 8: name = block3_conv2 , trainable = True
Layer 9: name = block3_conv3 , trainable = True
Layer 10: name = block3_pool , trainable = True
Layer 11: name = block4_conv1 , trainable = True
Layer 12: name = block4_conv2 , trainable = True
Layer 13: name = block4_conv3 , trainable = True
Layer 14: name = block4_pool , trainable = True
Layer 15: name = block5_conv1 , trainable = True
Layer 16: name = block5_conv2 , trainable = True
Layer 17: name = block5_conv3 , trainable = True
Layer 18: name = block5_pool , trainable = True
Layer 19: name = flatten , trainable = True
Layer 20: name = fc1 , trainable = True
Layer 21: name = fc2 , trainable = True

ทำการ freeze layer ทั้งหมด
```
vgg_extractor.trainable = False

for i,layer in enumerate(vgg_extractor.layers):  
    print( f"Layer {i}: name = {layer.name} , trainable = {layer.trainable}" )
```
Layer 0: name = input_3 , trainable = False
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
Layer 19: name = flatten , trainable = False
Layer 20: name = fc1 , trainable = False
Layer 21: name = fc2 , trainable = False

ทำการเพิ่ม dense layer สุดท้าย โดยมีการจำแนกข้อมูลเป็น 4 class เนื่องจาก เราต้องการทำนายรูปภาพขยะออกเป็น 4 ประเภท
```
x = vgg_extractor.output

# Add our custom layer(s) to the end of the existing model 

new_outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

# Construct the main model 
model = tf.keras.models.Model(inputs=vgg_extractor.inputs, outputs=new_outputs)
model.summary()
```
Model: "model_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param    
=================================================================
 input_3 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
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
                                                                 
 flatten (Flatten)           (None, 25088)             0         
                                                                 
 fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
 fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
 dense_3 (Dense)             (None, 4)                 16388     
                                                                 
=================================================================
Total params: 134,276,932
Trainable params: 16,388
Non-trainable params: 134,260,544

