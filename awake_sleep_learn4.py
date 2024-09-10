from keras import Model
from keras.src.applications.convnext import preprocess_input
from keras.src.applications.resnet import ResNet50
from keras.src.utils import img_to_array

from load_images import load_train_data, load_validation_data

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Flatten, Dense


def prepare_data(images, labels, boxes):
    X = []
    y_class = []
    y_box = []

    for img, label, box in zip(images, labels, boxes):
        img = img_to_array(img)
        img = preprocess_input(img)
        X.append(img)

        if label == 'none':
            y_class.append([1, 0, 0])  # none
            y_box.append([0, 0, 0, 0])  # no box
        elif label == 'awake':
            y_class.append([0, 1, 0])  # awake
            y_box.append(box)
        elif label == 'sleep':
            y_class.append([0, 0, 1])  # sleep
            y_box.append(box)

    X = np.array(X)
    y_class = np.array(y_class)
    y_box = np.array(y_box)

    return X, y_class, y_box


X_train, y_class_train, y_box_train = prepare_data(*load_train_data())
X_val, y_class_val, y_box_val = prepare_data(*load_validation_data())


def create_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Feature extractor
    model_fe = base_model(input_shape)

    # Region Proposal Network (RPN)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(model_fe)
    rpn_class = Conv2D(18, (1, 1), activation='sigmoid')(x)
    rpn_reg = Conv2D(36, (1, 1), activation='linear')(x)

    # Classification and Regression heads
    x = Flatten()(model_fe)
    x = Dense(1024, activation='relu')(x)
    class_pred = Dense(num_classes, activation='softmax')(x)
    box_pred = Dense(4, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=[rpn_class, rpn_reg, class_pred, box_pred])

    return model


input_shape = (224, 224, 3)
num_classes = 3  # none, awake, sleep
model = create_model(input_shape, num_classes)

model.compile(optimizer='adam',
              loss={'dense_2': 'categorical_crossentropy', 'dense_3': 'mean_squared_error'},
              loss_weights={'dense_2': 1.0, 'dense_3': 1.0},
              metrics={'dense_2': 'accuracy'})

model.fit(X_train, [y_class_train, y_box_train],
          validation_data=(X_val, [y_class_val, y_box_val]),
          epochs=10, batch_size=32)

model.save("awake_sleep_4.keras")

loss, class_loss, box_loss, class_acc = model.evaluate(X_val, [y_class_val, y_box_val])

print(f'Validation Loss: {loss}')
print(f'Classification Loss: {class_loss}')
print(f'Box Regression Loss: {box_loss}')
print(f'Classification Accuracy: {class_acc}')
