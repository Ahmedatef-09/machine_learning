import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10, 10, (observations,1))

generated_inputs = np.column_stack((xs,zs))
noise = np.random.uniform(-1, 1, (observations,1))
generated_targets = 2*xs - 3*zs + 5 + noise

np.savez('TF_Intro',inputs = generated_inputs,targets = generated_targets)
#solving with tensorflow
training_data = np.load('TF_Intro.npz')
# print(training_data)
input_size = 2 # x variable and z variable
output_size= 1 # y variable
#lets create our linear model 
model = tf.keras.Sequential([
        tf.keras.layers.Dense(output_size,
         kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
         bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))])
#optimize algorith and objevtive function
custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(optimizer = custom_optimizer,loss = 'mean_squared_error')
model.fit(training_data['inputs'],training_data['targets'],epochs= 100,verbose = 0)
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
# print(weights)
'''lets make some prediction '''
# plt.plot(model.predict_on_batch(training_data['inputs']),training_data['targets'])
x= model.predict_on_batch(training_data['inputs'])
plt.plot(x,training_data['targets'])
plt.show()
