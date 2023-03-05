<h1 align="center"> Image Classification </h1>

## A. Definisi Artificial Intelligence, Machine Learning, dan Deep Learning

<p align="center">
    <img src="https://user-images.githubusercontent.com/72246401/218114835-964ad3c9-72f5-4a3a-8d7c-e6940c9285dd.png" width="360" style="vertical-align:middle">
</p>

1.  Artificial Intelligence

    Dalam istilah sederhana, AI (Artificial Intelligence) ialah sistem atau mesin yang `meniru kecerdasan manusia` untuk melakukan tugas dan secara terus-menerus dapat meningkatkan kemampuannya berdasarkan `informasi yang telah dikumpulkan`. AI telah diwujudkan dalam beberapa bentuk. Beberapa contohnya adalah:
    
    <ol type="a">
    <li>Chatbot, menggunakan AI untuk memahami masalah pelanggan lebih cepat dan memberikan jawaban yang lebih efisien</li>
    <li>Smart Assistant, menggunakan AI untuk mengurai informasi penting dari kumpulan data teks dalam jumlah besar guna mengatur jadwal pengguna secara lebih baik</li>
    <li>Recommendation Engine, menggunakan AI supaya dapat memberikan rekomendasi otomatis untuk acara TV berdasarkan kebiasaan menonton pengguna, atau rekomendasi produk berdasarkan riwayat pembelian di platform ecommerce</li>
    </ol>

    Artificial Intelligence atau AI lebih banyak berkutat dengan proses mesin berpikir dan analisis data. Meskipun AI menampilkan gambar robot mirip manusia yang memiliki kemampuan berpikir cerdas, `AI tidak dimaksudkan untuk menggantikan manusia`.

    Ketika membicarakan AI, kita sebenarnya tidak jauh-jauh dari dua istilah ini yaitu `Machine Learning dan Deep Learning`. Karna keduanya merupakan sub-bidang dari Artificial Intelligence.

2.  Machine Learning

    Machine Learning yaitu istilah untuk menyebut komputer yang belajar dari data. Bidang ini merupakan irisan dari jurusan `ilmu komputer` dan `statistika` di mana algoritma digunakan untuk melakukan tugas tertentu tanpa diprogram secara eksplisit. Sebaliknya, `machine larning mengenali pola dalam data` dan membuat prediksi ketika ada data baru yang tiba.

    Secara umum, proses pembelajaran algoritma ini dapat bersifat `supervised ataupun unsupervised`, tergantung pada data yang digunakan untuk melatih algoritma. 

    Machine Learning menggabungkan `algoritma “klasik”` untuk berbagai macam tugas seperti `pengelompokan (clustering), regresi, atau klasifikasi`. Algoritma Machine Learning harus dilatih pada data dengan memberikan label tertentu. `Semakin banyak data` yang diberikan ke dalam algoritma, maka `hasilnya akan semakin baik`. 

    Machine learning sebenarnya adalah bagian yang lebih mengerucut dari bidang AI. Bidang ini menggunakan teknik (termasuk juga deep learning) yang memungkinkan mesin menggunakan pengalaman atau data sebelumnya untuk menyelesaikan tugas yang diberikan. Proses pembelajaran didasarkan pada langkah-langkah berikut:

    <ol type="a">
    <li>Masukkan data ke dalam algoritma. (Pada langkah ini kita dapat memberikan informasi tambahan ke model, misalnya, dengan melakukan ekstraksi fitur).</li>
    <li>Gunakan data tersebut untuk melatih model.</li>
    <li>Uji dan terapkan model.</li>
    <li>Gunakan model yang diterapkan untuk melakukan tugas prediksi otomatis. (Dengan kata lain, panggil dan gunakan model yang digunakan untuk menerima prediksi yang dikembalikan oleh model).</li>
    </ol>
    
3.  Deep Learning
    
    Deep Learning didefinisikan sebagai algoritma yang menganalisis data dengan struktur logika yang mirip dengan `cara manusia menarik kesimpulan`. Perlu diketahui bahwasanya deep learning dapat dicapai hanya dengan menggunakan supervised dan semisupervised learning.

    Untuk mencapai hal ini, sistem deep learning menggunakan struktur algoritma berlapis yang disebut `jaringan saraf tiruan (JST)` atau `Artificial Neural Network (ANN)` . Desain JST semacam itu terinspirasi oleh `jaringan saraf biologis otak manusia`, yang mengarah ke proses pembelajaran yang jauh lebih mumpuni daripada model machine learning standar.

    Saat ini, Deep Learning digunakan di banyak bidang. Sistem kemudi otomatis, misalnya, Deep learning digunakan untuk mendeteksi objek, seperti rambu STOP atau pejalan kaki. Militer menggunakan Deep learning untuk mengidentifikasi objek dari satelit, misal untuk menemukan zona aman atau tidak aman bagi pasukannya. Tentu saja, industri elektronik juga tak ketinggalan. Perangkat smart home assistant seperti Amazon Alexa, mengandalkan algoritma Deep Learning untuk merespons suara dan mengetahui preferensi penggunanya.
    

#### Perbedaan Machine Learning dan Deep Learning

<p align="center">
    <img src="https://user-images.githubusercontent.com/72246401/218115473-14f085dc-f1ec-4d06-ac17-b33ecb480eb4.png" width="480" style="vertical-align:middle">
</p>

Pendekatan klasifikasi secara Machine Learning (konvensional) umumnya melakukan ektraksi fitur secara terpisah kemudian dilanjutkan proses pembelajaran menggunakan metode klasifikasi konvensional.

Kelemahan pendekatan konvensional:
- Memerlukan waktu dan pengetahuan lebih untuk ekstraksi fitur.
- Sangat tergantung pada satu domain permasalahan saja sehingga tidak berlaku general.

Pendekatan klasifikasi berbasis Deep learning mempelajari representasi hirarki (pola fitur) secara otomatis melalui beberapa tahapan proses feature learning.

## B. Artificial Neural Network

#### Definisi Artificial Neural Network (ANN)

Salah satu metode mesin pembelajaran yang terinspirasi oleh cara kerja jaringan saraf biologis di otak manusia. Konsep ANN bermula pada artikel dari Waffen McCulloch dan Walter Pitts pada tahun 1943 yaitu mencoba untuk memformulasikan model matematis sel-sel otak manusia.

<p align="center">
    <img src="https://user-images.githubusercontent.com/72246401/218115848-72ea1fbc-f669-4926-a6c6-0f13164fa70e.png" alt="ANN/DNN" width="1080" style="vertical-align:middle">
</p>

#### Arsitektur Artificial Neural Network (ANN)

<p align="center">
    <img src="contents/ANN.gif" alt="ANN/DNN" width="510" style="vertical-align:left">
    <img src="contents/ANN weight.gif" width="480" style="vertical-align:left">
</p>

<br>Bagian-bagian Artificial Neural Network (ANN) / Dense Neural Network sebagai berikut:<br>

##### 1. Input Layer

Ukuran input sesuai dengan jumlah fitur pada data input.

```python
tf.keras.Input(shape=[height, width, color_channels])
```
```python
tf.keras.layers.InputLayer(input_shape=(height, width, color_channels))
```
```python
tf.keras.layers.Conv2D(input_shape=(height, width, color_channels))
```
```python
tf.keras.layers.Flatten(input_shape=(height, width, color_channels))
```
```python
tf.keras.layers.Dense(input_shape=(height, width, color_channels))
```
- input_shape/shape → dimensi ruang input

##### 2. Hidden Layer

Jumlah hidden layer sebaiknya disesuaikan dengan kompleksitas permasalahan.
- Semakin banyak jumlah layer memerlukan komputasi waktu yang lebih lama
- Semakin banyak jumlah node (neuron) memungkinkan mempelajari pola yang lebih rumit
- Untuk mencegah overfitting sebaiknya menambah jumlah node (neuron) secara bertahap

```python
tf.keras.layers.Dense(units, activation=None)
```
- units → dimensi ruang output
- activation → fungsi aktivasi untuk digunakan → relu
  
##### 3. Output Layer

Jumlah neuron sesuai dengan permasalahan.
- Untuk `klasifikasi binary dan regresi` menggunakan `satu neuron`.
- Untuk `klasifikasi multiclass atau categorical` menggunakan `jumlah neuron sesuai jumlah kelas`.

```python
tf.keras.layers.Dense(units, activation=None)
```
- units → dimensi ruang output
  - binary → satu neuron.
  - categorical → jumlah neuron sesuai jumlah kelas.
- activation → fungsi aktivasi untuk digunakan
  
  | activation  | output/class mode | loss function | 
  |     ---     |        ---        |     ---       |
  |   sigmoid   |      binary       |  binary_crossentropy → 0/1 |
  |   softmax   |   categorical     |     categorical_crossentropy → [1 0] [0 1]|
  |   softmax   |   categorical     |  sparse_categorical_crossentropy → [0] [1] |

## C. Convolutional Neural Network

#### Definisi Convolutional Neural Network

Convolutional Neural Network (CNN/ConvNet) merupakan bagian dari deep neural network, yakni jenis jaringan saraf tiruan yang umumnya digunakan dalam pengenalan dan pemrosesan gambar. Algoritma ini memiliki dua bagian utama yaitu bagian ekstraksi fitur dan bagian klasifikasi. Bagian ekstraksi fitur gambar dilakukan oleh Convolutional Backbone, sedangkan untuk bagian klasifikasi dilakukan oleh Classifier Head yang berisi Artificial Neural Network.

<p align="center">
    <img src="https://user-images.githubusercontent.com/72246401/218117099-2ecca696-9e1c-44d2-98f2-b97f29b83016.png" alt="overview cnn" width="480" style="vertical-align:middle">
</p>

#### Arsitektur Convolutional Neural Network

##### 1. Convolution Layer

Convolutional layer merupakan proses konvolusi citra input dengan filter yang menghasilkan `feature map` (fitur-fitur penting sebuah gambar memberikan informasi jauh lebih besar dibanding gambar input itu sendiri).

<p align="center">
    <img src="contents/conv backbone.gif" width="480" style="vertical-align:middle">
</p>

Proses konvolusi citra dengan filter dilakukan `sliding filter` mulai dari kiri atas dari matrik citra sampai kanan bawah.

<p align="center">
    <img src="contents/conv.gif" alt="convolution" width="720" style="vertical-align:left">
</p>

Convolutional layer akan mencari kernel/weight yang tepat. Semakin banyak kernel maka semakin banyak jenis fitur. Tujuan Convolutional layer ditumpuk agar informasi yang didapatkan semakin lebih bermakna.

<p align="center">
    <img src="contents/conv stride 1 pad 0.gif" alt="convolution" width="360" style="vertical-align:left">
    <img src="contents/conv stride 1 pad 1.gif" alt="convolution" width="360" style="vertical-align:left">
</p>
<p align="center">
    <img src="contents/conv stride 2 pad 0.gif" alt="convolution" width="360" style="vertical-align:left">
    <img src="contents/conv stride 2 pad 1.gif" alt="convolution" width="360" style="vertical-align:left">
</p>

```python
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    activation=None,
    input_shape=(height, width, color_channels)
)
```

- filters → dimensi ruang output → jumlah filter output dalam konvolusi
- kernel_size → ukuran spasial dari filter (lebar/tinggi)
- stride → besar pergeseran filter dalam konvolusi
- padding → jumlah penambahan nol pada gambar
    - valid → tidak ada padding
    - same → padding nol merata kiri/kanan/atas/bawah
- activation → fungsi aktivasi untuk digunakan
- input_shape → input gambar

##### 2. Batch Normalization
Batch Normalization berperan untuk mengurangi pergeseran kovarian atau menyamakan distribusi setiap nilai input yang selalau berubah karena perubahan pada layer sebelumnya selama proses training.

<p align="center">
    <img src="contents/batchnorm.png" width="480" style="vertical-align:left">
</p>

```python
tf.keras.layers.BatchNormalization()
```

##### 3. Pooling Layer

Pooling layer berperan untuk memperkecil dimensi feature image (downsampling) dan menyimpan informasi penting.

<p align="center">
    <img src="contents/pooling.png" alt="pooling" width="640" style="vertical align:middle">
</p>

<p align="center">
    <img src="contents/max pol stride 1 pad 1.gif" alt="convolution" width="360" style="vertical-align:left">
    <img src="contents/max pol stride 2 pad 1.gif" alt="convolution" width="360" style="vertical-align:left">
</p>

```python
tf.keras.layers.MaxPool2D(
    pool_size=(2, 2),
    strides=None,
    padding='valid',
)
```

```python
tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=None,
    padding='valid',
)
```

- pool_size → ukuran pool
- strides → besar pergeseran
- padding → jumlah penambahan nol pada gambar
    - valid → tidak ada padding
    - same → padding nol merata kiri/kanan/atas/bawah

##### Visualisasi Proses Konvolusi dan Pooling pada Gambar

Berikut adalah salah satu contoh bagaimana gambar anjing diproses setelah melewati Convolutional Backbone yang terdiri dari beberapa convolution layer dan max pooling layer. Terlihat pada gambar, fitur-fitur penting pada wajah anjing akan semakin muncul setelah dilakukan proses konvolusi dan pooling. Proses ini akan mempermudah model dalam mengenali gambar yang akan dilatih dan diprediksi nantinya.

<p align="center">
    <img src="contents/convolution visualization.png" style="vertical align:middle">
</p>

#### Classifier Head (ANN)

<p align="center">
    <img src="contents/ANN.gif"  width="480" style="vertical align:middle">
</p>

##### 1. Flatten dan Global Pooling

Flatten dan Global Pooling berperan sebagai `input layer`.

<p align="center">
    <img src="contents/fully connected layer vs global average pooling.png" alt="flatten vs global average pooling" width="640" style="vertical align:middle">
</p>

- Flatten → hxwxd (1 dimension)
  ```python
  tf.keras.layers.Flatten()
  ```
- Global Pooling -> d (1 dimension)
  ```python
  tf.keras.layers.GlobalAveragePooling2D()
  ```
  ```python
  tf.keras.layers.GlobalMaxPool2D()
  ```

##### 2. Hidden Layer

```python
tf.keras.layers.Dense(units, activation=None)
```
- units → dimensi ruang output
- activation → fungsi aktivasi untuk digunakan → relu
  
  
##### 3. Output Layer

Jumlah neuron sesuai dengan permasalahan.
- Untuk `klasifikasi binary dan regresi` menggunakan `satu neuron`.
- Untuk `klasifikasi multiclass atau categorical` menggunakan `jumlah neuron sesuai jumlah kelas`.

```python
tf.keras.layers.Dense(units, activation=None)
```
- units → dimensi ruang output
  - binary → satu neuron.
  - categorical → jumlah neuron sesuai jumlah kelas.
- activation → fungsi aktivasi untuk digunakan
  
  | activation  | output/class mode | loss function | 
  |     ---     |        ---        |     ---       |
  |   sigmoid   |      binary       |  binary_crossentropy → 0/1 |
  |   softmax   |   categorical     |     categorical_crossentropy → [1 0] [0 1]|
  |   softmax   |   categorical     |  sparse_categorical_crossentropy → [0] [1] |

## D. Strategi Proses Training

<p align="center">
    <img src="contents/Strategi Proses Pembelajaran.png" alt="Stratego Proses Pembelajaran" width="640" style="vertical align:middle">
</p>

#### 1. Modifikasi Network
- Merubah arsitektur, misalnya menambah jumlah hidden layer, jumlah neuron, atau jenis arsitektur lain.
  <p align="center">
    <img src="contents/CNN architecture.ppm" width="540" style="vertical-align:left">
  </p>
- Merubah fungsi aktivasi, misalnya menggunakan ReLU (apabila x ≤ 0 maka x = 0 dan apabila x > 0 maka x = x).
  <p align="center">
    <img src="contents/Fungsi Aktivasi.gif" width="480" style="vertical-align:left">
  </p>

#### 2. [Optimasi parameter](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers?version=nightly)
Nilai learning rate berpengaruh pada perhitungan bobot baru, umumnya penggunaan learning rate yang menyesuaikan nilai gradien (adaptive learning rate) menunjukkan kinerja model yang lebih baik. Contoh algoritma adaptive learning rate seperti Adagrad, Adadelta, Adam, AdaSecant, dan RMSprop.

#### 3. Mencegah Overfitting

- Regularisasi dilakukan untuk mengurangi generalization error dengan mencegah model lebih 
kompleks.
    - [Regularization L1 norm dan Regularization L2](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers) norm (weight decay)
    
- [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) adalah proses mencegah terjadinya overfitting dan juga mempercepat proses learning.
  
    <p align="center">
      <img src="contents/dropout.jpg" alt="dropout" width="640" style="vertical-align:middle">
    </p>

    ```python
    tf.keras.layers.Dropout(rate)
    ```
- [Early Stopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) adalah iterasi pada saat training dihentikan jika `generalization error/loss validation` mulai naik.

- Augmentasi Data adalah menambah data training. Augmentasi data menggunakan [Image Data Generator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator).

---

- Visualisasi: [ANN](https://playground.tensorflow.org/)
- Visualisasi: [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- Code: [Malaria](https://colab.research.google.com/drive/1jQ3NObFqafI3jWGLDTV1Ca8bXIGPVlL_?usp=sharing)
- [Pengumpulan Magang 1](https://forms.gle/DceSF85ZPad1Jj3Y6)
