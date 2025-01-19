import os
import sys

import numpy as np
import tensorflow as tf


def log(msg: str):
    print("[" + os.path.basename(__file__) + "] " + msg)


if __name__ == '__main__':
    input_image_path = None

    # 如果参数只有 1 个，说明是在 Pycharm 等编辑器里面跑的，或者在命令行中没有传入参数。
    if len(sys.argv) == 1:
        log("argv == 1, debug")
        input_image_path = "C:/Workspace/Github/python-demos/tensorflow/classify/dataset/4-home/img_580.png"
    else:
        input_image_path = sys.argv[1]

    img_height = 180
    img_width = 180

    img = tf.keras.utils.load_img(
        input_image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    current_file_path = os.path.dirname(__file__)
    production_dir = os.path.join(current_file_path, "production")
    if not os.path.exists(production_dir):
        log("production 目录不存在，请先通过 train.py 训练自己的模型")
        exit(1)

    model_path = os.path.join(production_dir, 'model.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # 打印输入输出向量的名称
    interpreter.get_signature_list()

    # 这里的 serving_default 是上面一行打印得到的
    classify_lite = interpreter.get_signature_runner('serving_default')
    # 这里的 sequential_1_input 也是打印得到的
    predictions_lite = classify_lite(rescaling_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)

    # 读取分类文件
    class_names_file_path = os.path.join(production_dir, "class_names.txt")
    lines = open(class_names_file_path)
    class_names = []
    line = lines.readline()
    while line:
        class_names.append(line.strip())
        line = lines.readline()

    # 类别
    class_type = class_names[np.argmax(score_lite)]

    # 置信度
    confidence = 100 * np.max(score_lite)

    log("This image most likely belongs to \"{}\" with a {:.2f} percent confidence.".format(class_type, confidence))
