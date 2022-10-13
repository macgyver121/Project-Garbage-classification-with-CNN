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

## EDA
### Import data
ทำการ import image dataset จาก google drive

'''
data_dir = '/content/drive/MyDrive/hw2_DADS7202_photo_4class'

np.random.seed(1234)
tf.random.set_seed(5678)

# Defining data generator withour Data Augmentation
data_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale = 1/255., validation_split = 0.3)

train_data = data_gen.flow_from_directory(data_dir, 
                                          target_size = (224, 224), 
                                          batch_size = ....,
                                          subset = 'training',
                                          class_mode = 'binary')
test_data = data_gen.flow_from_directory(data_dir, 
                                        target_size = (224, 224), 
                                        batch_size = ....,
                                        subset = 'testing',
                                        class_mode = 'binary')
'''

- ทำการ split data เป็น train และ test สัดส่วน 70:30
- rescale รูปภาพเป็น 224*224 pixel
- batch size = ..........



