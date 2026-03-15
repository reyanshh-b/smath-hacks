import argparse, os, json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Root dir with train/val subfolders')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', default='models/model.h5')
    args = parser.parse_args()
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    train_ds = image_dataset_from_directory(train_dir, image_size=(224,224), batch_size=args.batch)
    val_ds = image_dataset_from_directory(val_dir, image_size=(224,224), batch_size=args.batch)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print('Classes:', class_names)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    model = build_model(num_classes)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    # write class names for the app to read
    with open(os.path.join(os.path.dirname(args.out), 'class_names.txt'), 'w') as f:
        for c in class_names:
            f.write(c + '\\n')
    print('Saved model to', args.out)
    print('Saved class names to', os.path.join(os.path.dirname(args.out), 'class_names.txt'))

if __name__ == '__main__':
    main()
