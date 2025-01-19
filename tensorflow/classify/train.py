# 这个文件，需要运行在有 tensorflow 的环境中，
# 例如在 Windows 电脑上，需要在 Pycharm 中，选择 Setting > Project: classify > Python Interpreter > Add Interpreter 添加或
# 者选择含有 tensorflow 的环境。
# pip install tensorflow
# pip install ffmpeg-python
import os.path

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def log(msg: str):
    print("[" + os.path.basename(__file__) + "] " + msg)


if __name__ == '__main__':
    log("[main] begin")

    # 下载原始素材
    import pathlib
    current_file_path = os.path.dirname(__file__)
    training_dir = os.path.join(current_file_path, "dataset")
    data_dir = training_dir
    log("即将从 " + data_dir + " 中获取所有的训练图片。")

    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    log("总共有 " + str(image_count) + " 张照片。")

    batch_size = 32
    img_height = 180
    img_width = 180

    # 训练集
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # 测试集
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(class_names)

    # 设置缓存提高性能
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 构建模型
    num_classes = len(class_names)
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # compile and summary
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    # 训练模型
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 可视化训练过程
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # 转化为 TensorFlow Lite 的模型文件并保存
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # 产物保存目录
    production_dir = os.path.join(current_file_path, "production")
    if not os.path.exists(production_dir):
        os.mkdir(production_dir)

    # 保存模型
    model_save_path = os.path.join(production_dir, "model.tflite")
    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)

    # 保存分类名
    class_names_path = os.path.join(production_dir, "class_names.txt")
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(str(name))
            f.write("\n")
    log("[main] end")
