# Kütüphaneleri import ediyoruz
import numpy as np # numpy kütüphanesi
import pandas as pd # pandas kütüphanesi

# tensorflow kütüphanesini uç kütüphane olarak ayarlıyoruz
from keras import backend as K
# K.set_image_dim_ordering('tf') --> K.set_image_data_format('channels_last')
# K.set_image_dim_ordering('tf') orjinal koddaki hali
K.set_image_data_format('channels_last') #yeni versiyonlarda değişiklik yapılmış hali 

# Grafikle gösterim işlemleri için matplotlib kütüphanesini ekliyoruz
import matplotlib.pyplot as plt
# %matplotlib inline

#cuda hatası çözümü
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# train.csv'deki verileri değişkene atıyoruz.
train_data = pd.read_csv("../dataset/train.csv")
# train_data değişkeninin boyutlarını kontrol ediyoruz.
print(train_data.shape)

# Modeli eğitirken resimlere ve etiketlere ihtiyacımız var.
# train_data içerisindeki verileri images ve labels olarak ayırıyoruz.
images = train_data.iloc[:, 1:]
labels = train_data.iloc[:, 0]

# as_matrix metodu 0.23.0 versiyonundandan sonra kaldırılmıştır.
# labels = labels.as_matrix()
# images = images.as_matrix().reshape(images.shape[0], 28, 28, 1)
# ayırdığımız verileri CNN'de kullanabilmek için numpy array haline getiriyoruz.
# images içerisindeki verileri CNN'de kullanabilmek için boyut ayarlaması yapıyoruz.
labels = labels.to_numpy()
images = images.to_numpy().reshape(images.shape[0], 28, 28, 1)

# Öznitelikleri normalize eden fonksiyon
def normalize_grayscale(image_data):
    # Resimleri [0.1, 0.9] aralığında minumum-maksimum ölçeklemesi yapıyoruz.
    return (25.5 + 0.8 * image_data) / 255
#train_features değişkenindeki verileri normalize fonksiyonuna gönderiyoruz.
train_features = normalize_grayscale(images)

# Etiket değerleri için one-hot encoding uyguluyoruz.
from keras.utils import np_utils
train_labels = np_utils.to_categorical(labels)

# Verimizi eğitim ve doğrulama olarak ikiye ayırıyoruz
# Doğrulama verimiz toplam verimizin %15i olarak ayarlıyoruz
from sklearn.model_selection import train_test_split
train_features, val_features, train_labels, val_labels = train_test_split(train_features, 
                                                                          train_labels,
                                                                          test_size=0.15,
                                                                          random_state=np.random.randint(300))

# Oluşturduğumuz verilerin boyutlarını kontrol ediyoruz. 
print('train_features shape: ', train_features.shape)
print('val_features shape: ', val_features.shape)
print('train_labels shape: ', train_labels.shape)
print('val_labels shape: ', val_labels.shape)

# Kerasta model ve katmanları oluşturabilmek için gerekli kütüphaneler
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout
from keras.models import Model, load_model

# Keras kütüphanesini kullanarak modelimizi tanımlıyoruz.
def get_model(input_shape):
    drop = 0.3
    X_input = Input(input_shape)
    X = Conv2D(64, (5,5), strides=(1,1), activation='relu', 
               kernel_initializer='glorot_normal')(X_input)
    X = MaxPooling2D((2,2))(X)
    X = Conv2D(128, (5,5), strides=(1,1), activation='relu',
              kernel_initializer='glorot_normal')(X)
    X = MaxPooling2D((2,2))(X)
    X = Flatten()(X)
    X = Dense(256, activation='relu')(X)
    X = Dropout(drop)(X)
    X = Dense(32, activation='relu')(X)
    X = Dropout(drop)(X)
    X = Dense(10, activation='softmax')(X)
    model = Model(inputs=[X_input], outputs=[X])
    return model

# Optimizer için kütüphane 
from keras.optimizers import Nadam
# Optimizerimizi tanımlıyoruz ve learning rate 0.001 olarak ayarlıyoruz.
opt = Nadam(lr=0.001)
# Model fonksiyonumuza resmin giriş boyutlarını gönderiyoruz.
model = get_model((28, 28, 1))
# Modelimizi derliyoruz.
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modelimizin tüm parametrelerini incelemek için kullanıyoruz.
model.summary()

# Model kontrol noktası oluşturmak için keras kütüphanesi 
from keras.callbacks import ModelCheckpoint
# Modelimizin kayıt biçimi tanımlanıyor.
f_path = 'model.h5'
# Modelimizin eğitimdeki en iyi verisi model.h5 olarak kaydedilecek
msave = ModelCheckpoint(f_path, save_best_only=True)

# Verimizi eğittiğimiz kısım 
epochs = 5 # epochs değerini 5 olarak belirliyoruz.
batch_size = 64 # her seferinde okunacak yığın miktarını 64 olarak belirliyoruz.
training = model.fit(train_features, train_labels,
                     validation_data=(val_features, val_labels),
                     epochs=epochs,
                     callbacks=[msave],
                     batch_size=batch_size, 
                     verbose=1)

# Eğitimdeki verileri history metoduyla alıyoruz.
loss = training.history['loss'] # kayıp değeri
val_loss = training.history['val_loss'] # doğrulama kayıp değeri
acc = training.history['accuracy'] # doğruluk değeri
val_acc = training.history['val_accuracy'] # doğrulama doğruluk değeri
# Kayıp değerini çizdiriyoruz
tra = plt.plot(loss)
val = plt.plot(val_loss, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(["Training", "Validation"])
plt.show()
# Doğruluk değerini çizdiriyoruz
plt.plot(acc)
plt.plot(val_acc, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Accuracy')
plt.legend(['Training', 'Validation'], loc=4)
plt.show()

# Kaydettiğimiz en iyi modeli yüklüyoruz 
model = load_model(f_path)
# Test verilerini yüklüyoruz
test_data = pd.read_csv('../dataset/test.csv')
# Test verilerini numpy array çevirip boyutunu düzenliyoruz
test_images = test_data.to_numpy().reshape(test_data.shape[0], 28, 28, 1)
# Test verilerine normalizasyon yapıyoruz
test_features = normalize_grayscale(test_images)
# Test verilerimize tahmin işlemi yapıyoruz
pred = model.predict(test_features, batch_size=batch_size, 
                       verbose=1)
# Tahmin değerlerimizi 0...9 arası kategorik veriye çeviriyoruz.
pred_digits = np.argmax(pred, axis=1)
# Tahmin verilerimizi csv dosyası olarak proje klasörüne kaydediyoruz.
submission = pd.DataFrame({'Label': pred_digits})
submission.index += 1
submission.index.name = "ImageId"
submission.to_csv('submission.csv')

