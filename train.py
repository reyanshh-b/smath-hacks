"""
CoralScan — Python Offline Trainer
====================================
For devs who want to train a more powerful model using a full dataset.
The trained model is saved as both .h5 (for Python/Flask) and also
exported to TF.js format so it can be loaded in the browser.

Usage:
  python train.py --data-dir data --epochs 30 --export-tfjs

Data directory structure:
  data/
    healthy/
      img1.jpg
      img2.jpg
      ... 
    unhealthy/
      img1.jpg
      img2.jpg
      ...

Or train/val split:
  data/
    train/
      healthy/
      unhealthy/
    val/
      healthy/
      unhealthy/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF startup noise

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMG_SIZE   = (224, 224)
CLASSES    = ['healthy', 'unhealthy']

def parse_args():
    p = argparse.ArgumentParser(description='CoralScan offline trainer')
    p.add_argument('--data-dir',   default='data',           help='Root data directory')
    p.add_argument('--output-dir', default='models',         help='Where to save models')
    p.add_argument('--epochs',     type=int, default=30,     help='Training epochs')
    p.add_argument('--batch-size', type=int, default=16,     help='Batch size')
    p.add_argument('--lr',         type=float, default=1e-4, help='Learning rate')
    p.add_argument('--val-split',  type=float, default=0.2,  help='Validation split (if no val/ folder)')
    p.add_argument('--export-tfjs', action='store_true',     help='Export to TF.js format for browser use')
    return p.parse_args()


def build_generators(data_dir, batch_size, val_split):
    has_split = (
        os.path.isdir(os.path.join(data_dir, 'train')) and
        os.path.isdir(os.path.join(data_dir, 'val'))
    )

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.75, 1.25],
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.0 if has_split else val_split,
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    kwargs = dict(target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical', shuffle=True)

    if has_split:
        print(f'Using train/val split folders from {data_dir}/')
        train_gen = train_aug.flow_from_directory(os.path.join(data_dir, 'train'), **kwargs)
        val_gen   = val_aug.flow_from_directory(os.path.join(data_dir, 'val'), shuffle=False,
                                                 target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical')
    else:
        print(f'Using {val_split*100:.0f}% validation split from {data_dir}/')
        train_gen = train_aug.flow_from_directory(data_dir, subset='training',   **kwargs)
        val_gen   = ImageDataGenerator(rescale=1.0/255, validation_split=val_split).flow_from_directory(
            data_dir, subset='validation', shuffle=False,
            target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical'
        )

    return train_gen, val_gen


def build_model():
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax'),
    ])

    return model, base


def get_class_weights(gen):
    counts = np.bincount(gen.classes)
    total  = gen.samples
    return {i: total / (len(counts) * c) for i, c in enumerate(counts)}


def plot_history(history, ft_history, output_dir):
    acc     = history.history['accuracy']     + ft_history.history['accuracy']
    val_acc = history.history['val_accuracy'] + ft_history.history['val_accuracy']
    loss    = history.history['loss']         + ft_history.history['loss']
    val_loss= history.history['val_loss']     + ft_history.history['val_loss']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#050f1a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0a1a2a')
        ax.tick_params(colors='#5a8aa8')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a3050')

    epochs_range = range(len(acc))
    ax1.plot(epochs_range, acc,     color='#00ccff', label='Train')
    ax1.plot(epochs_range, val_acc, color='#00e676', linestyle='--', label='Val')
    ax1.axvline(x=len(history.history['accuracy'])-1, color='#ff9800', linewidth=1, linestyle=':', label='Fine-tune start')
    ax1.set_title('Accuracy', color='#c0dff0', fontsize=11)
    ax1.set_xlabel('Epoch', color='#5a8aa8')
    ax1.legend(labelcolor='#c0dff0', framealpha=0)

    ax2.plot(epochs_range, loss,     color='#00ccff', label='Train')
    ax2.plot(epochs_range, val_loss, color='#00e676', linestyle='--', label='Val')
    ax2.axvline(x=len(history.history['loss'])-1, color='#ff9800', linewidth=1, linestyle=':', label='Fine-tune start')
    ax2.set_title('Loss', color='#c0dff0', fontsize=11)
    ax2.set_xlabel('Epoch', color='#5a8aa8')
    ax2.legend(labelcolor='#c0dff0', framealpha=0)

    plt.tight_layout()
    out = os.path.join(output_dir, 'training_plot.png')
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#050f1a')
    print(f'Training plot saved to {out}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('\n' + '='*55)
    print(' CoralScan Offline Trainer')
    print('='*55)
    print(f' Data dir    : {args.data_dir}')
    print(f' Output dir  : {args.output_dir}')
    print(f' Epochs      : {args.epochs}')
    print(f' Batch size  : {args.batch_size}')
    print(f' Learning rate: {args.lr}')
    print('='*55 + '\n')

    # Check data dir
    if not os.path.isdir(args.data_dir):
        print(f'ERROR: Data directory "{args.data_dir}" not found.')
        print('Create it with subfolders: healthy/ and unhealthy/')
        sys.exit(1)

    # Data generators
    train_gen, val_gen = build_generators(args.data_dir, args.batch_size, args.val_split)
    print(f'\nTraining samples  : {train_gen.samples}')
    print(f'Validation samples: {val_gen.samples}')
    print(f'Class indices     : {train_gen.class_indices}')

    if train_gen.samples < 4:
        print('ERROR: Not enough training images. Need at least 4 total.')
        sys.exit(1)

    class_weights = get_class_weights(train_gen)
    print(f'Class weights     : {class_weights}')

    # Build model
    model, base = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    model_path = os.path.join(args.output_dir, 'model.keras')
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ]

    # Phase 1: train head only
    print('\n--- Phase 1: Training head layers ---')
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Phase 2: fine-tune top layers
    print('\n--- Phase 2: Fine-tuning top 40 base layers ---')
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr * 0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    ft_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Save final model
    model.save(model_path)
    model.save(os.path.join(args.output_dir, 'model.h5'))
    with open(os.path.join(args.output_dir, 'class_names.txt'), 'w') as f:
        for cls in CLASSES:
            f.write(cls + '\n')

    best_val = max(history.history['val_accuracy'] + ft_history.history['val_accuracy'])
    print(f'\nBest val accuracy : {best_val:.4f} ({best_val*100:.1f}%)')
    print(f'Model saved to    : {model_path}')

    # Export to TF.js
    if args.export_tfjs:
        try:
            import tensorflowjs as tfjs
            tfjs_path = os.path.join(args.output_dir, 'tfjs_model')
            tfjs.converters.save_keras_model(model, tfjs_path)
            print(f'TF.js model exported to: {tfjs_path}')
            print('To use in the browser, copy tfjs_model/ to your static/ folder')
            print('and load with: tf.loadLayersModel("static/tfjs_model/model.json")')
        except ImportError:
            print('tensorflowjs not installed. Run: pip install tensorflowjs')
            print('Then re-run with --export-tfjs to get a browser-loadable model.')

    plot_history(history, ft_history, args.output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
