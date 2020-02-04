
from kersa.layers import Input,GRU,Dense

observations = Input(shape=(n_steps,observation_dim),dtype="float32")

latent_dim = 64
latent_rnn = GRU(latent_dim,return_sequences=True)
latent_to_action = Dense(action_dim)

output1 = layer1(inputs)
output = layer2(output1)

model = keras.models.Model(inputs = inputs, outputs = output)
model.compile("adam",loss="mse")
