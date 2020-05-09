import tensorflow as tf
import tensorflow_datasets as tfds
#tfds.disable_progress_bar()

mirrored_strategy = tf.distribute.MirroredStrategy()

def get_data():
  datasets, ds_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']

  BUFFER_SIZE = 10000
  BATCH_SIZE_PER_REPLICA = 64
  BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

  return train_dataset, eval_dataset

def get_model():
  with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

#kears-API
def k_model(mode = 'restored',save = 0):
    model = get_model()
    train_dataset, eval_dataset = get_data()
    keras_model_path = "./keras_save"
    if mode =='train':
        model.fit(train_dataset, epochs=2)
        model.save(keras_model_path)  #总是建议用keras API还原
    else:
        if save == 0:#keras API还原
            restored_keras_model = tf.keras.models.load_model(keras_model_path)
            restored_keras_model.fit(train_dataset, epochs=2) #还原模型，继续训练
        elif save == 1: #keras API分布式还原
            another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
            with another_strategy.scope():
                restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
                restored_keras_model_ds.fit(train_dataset, epochs=2)
        else : #低级API还原
            another_strategy = tf.distribute.MirroredStrategy()
            # Loading the model using lower level API
            with another_strategy.scope():
            loaded = tf.saved_model.load(keras_model_path)
#tf.saved_model API

if __name__=='__main__':
    k_model()

