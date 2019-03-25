# Machine Learning with Tensorflow

## Regression Dense Neural Network Model (Celcius to Fahrenheit Conversion)
- ML Terms:
  - **Feature**: The input(s) to our model
  - **Examples**: An input/output pair used for training
  - **Labels**: The output of the model
  - **Layer**: A collection of nodes connected together within a neural network.
  - **Model**: The representation of your neural network
  - **Dense and Fully Connected (FC)**: Each node in one layer is connected to each node in the previous layer.
  - **Weights and biases**: The internal variables of model
  - **Loss**: The discrepancy between the desired output and the actual output
  - **MSE**: Mean squared error, a type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.
  - **Gradient Descent**: An algorithm that changes the internal variables a bit at a time to gradually reduce the loss function.
  - **Optimizer**: A specific implementation of the gradient descent algorithm. (There are many algorithms for this. In this course we will only use the “Adam” Optimizer, which stands for ADAptive with Momentum. It is considered the best-practice optimizer.)
  - **Learning rate**: The “step size” for loss improvement during gradient descent.
  - **Batch**: The set of examples used during training of the neural network
  - **Epoch**: A full pass over the entire training dataset
  - **Forward pass**: The computation of output values from input
  - **Backward pass (backpropagation)**: The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.
- Model Training:
  ```python
  # tf.__version__ = 1.13.1
  import tensorflow as tf
  l0 = tf.keras.layers.Dense(units=1, input_shape=[1]) 
  model = tf.keras.Sequential([l0])
  model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
  history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
  model.predict([100.0])
  ```
  
  ```python
  hidden = tf.keras.layers.Dense(units=2, input_shape=[3])
  output = tf.keras.layers.Dense(units=1)
  model = tf.keras.Sequential([hidden, output])
  ```

## Classification Dense Neural Network Model (Fashion MNIST)
- ML Terms:
  - **Flattening** is the process of converting a 2D image into 1D vector.
  - **ReLU (Rectified Linear Unit)**: An activation function that allows a model to solve non-linear problem.
  - **Softmax**: A function that provides probabilities for each possible output class.
  - **Regression**: A model that outputs a single value. For example, an estimate of a house’s value.
  - **Classification**: A machine learning model used for distinguishing among two or more output categories.
- Regression vs. Classification:
  
  ||Classification|Regression|
  |:-:|:-:|:-:|
  |Output|List of numbers that represent probabilities for each class|Single Number|
  |Example|Fashion MNIST|Celcius to Fahrenheit|
  |Loss|Sparse Categorical Cross Entropy|Mean Squared Error|
  |Last Layer Activation Function|Softmax|None|
- Model Training:  
  ```python
  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # Flattening: input image 28x28=784 pixels
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu), # ReLU
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax) # Softmax
    ])
  
  model.compile(optimizer='adam',
    loss='sparce_categorical_crossentropy',
    metrics=['accuracy'])
  
  # Normalize the dataset
  def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

  # The map function applies the normalize function to each element in the train and test datasets
  train_dataset =  train_dataset.map(normalize)
  test_dataset  =  test_dataset.map(normalize)
  
  NUM_EXAMPLES = 60000
  BATCH_SIZE = 32
  train_dataset = train_dataset.repeat().shuffle(NUM_EXAMPLES).batch(BATCH_SIZE)
  test_dataset = test_dataset.batch(BATCH_SIZE)
  
  model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))
  
  test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_train_examples/BATCH_SIZE))
  ```

## Convolution Neural Network (Fashion MNIST)
- Main concepts:
  - Convolution.
  - Max Pooling.
- **Convolution**: 
