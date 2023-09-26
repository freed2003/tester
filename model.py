"""
    Place model here
    Args:
        -orig_data: origintal time-series data
    Output:
        Generated data
"""
# import torch

# def model(ori_data):
#     model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')




import tensorflow as tf
import keras
from keras import layers
import numpy as np
from utils import extract_time, random_generator, batch_generator


def model (ori_data, parameters):
    hidden_dim   = parameters['hidden_dim'] 
    # num_layers   = parameters['num_layer']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    z_dim        = 2
    gamma        = 1
    ori_time, max_seq_len = extract_time(ori_data)
    no, seq_len, dim = np.asarray(ori_data).shape
    def MinMaxScaler(data):
        """Min-Max Normalizer.
        
        Args:
        - data: raw data
        
        Returns:
        - norm_data: normalized data
        - min_val: minimum values (for renormalization)
        - max_val: maximum values (for renormalization)
        """    
        min_val = np.min(np.min(data, axis = 0), axis = 0)
        data = data - min_val
      
        max_val = np.max(np.max(data, axis = 0), axis = 0)
        norm_data = data / (max_val + 1e-7)
        
        return norm_data, min_val, max_val
    ori_data, min_val, max_val = MinMaxScaler(ori_data)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mean_squared = tf.keras.losses.MeanSquaredError()
    def embedder():
        model = keras.Sequential()
        model.add(layers.GRU(hidden_dim))
        model.add(layers.Dense(hidden_dim, activation="sigmoid"))
        return model
    def recovery():
        model = keras.Sequential()
        model.add(layers.GRU(hidden_dim))
        model.add(layers.Dense(dim, activation="sigmoid"))
        return model
    def generator():
        model = keras.Sequential()
        model.add(layers.GRU(hidden_dim))
        model.add(layers.Dense(hidden_dim, activation="sigmoid"))
        return model
    def supervisor():
        model = keras.Sequential()
        model.add(layers.GRU(hidden_dim))
        model.add(layers.Dense(hidden_dim, activation="sigmoid"))
        return model
    def discriminator():
        model = keras.Sequential()
        model.add(layers.GRU(hidden_dim))
        model.add(layers.Dense(1))
        return model
    
    emb = embedder()
    rec = recovery()
    gen = generator()
    sup = supervisor()
    disc = discriminator()

    E0_solver = tf.keras.optimizers.Adam(1e-4)
    E_solver = tf.keras.optimizers.Adam(1e-4)
    D_solver = tf.keras.optimizers.Adam(1e-4)
    G_solver = tf.keras.optimizers.Adam(1e-4)   
    GS_solver = tf.keras.optimizers.Adam(1e-4)
    print('Start Embedding Network Training')
    for itt in range(iterations):
        X_mb = batch_generator(ori_data, batch_size)
        print("Shape: ", X_mb.shape)
        H = emb(X_mb)
        X_tilde = rec(H)
        E_loss_T0 = mean_squared(X_mb, X_tilde)
        E_loss0 = 10*tf.sqrt(E_loss_T0)
        E0_solver.minimize(E_loss0, rec.trainable_variables + emb.trainable_variables)

        if itt % 1000 == 0:
            print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(E_loss_T0),4)) ) 
    print('Finish Embedding Network Training')


    print('Start Training with Supervised Loss Only')
        
    for itt in range(iterations):
        X_mb = batch_generator(ori_data, batch_size)      
        Z_mb = random_generator(batch_size, z_dim, max_seq_len)    
        H = embedder(X_mb)
        H_hat_supervise = supervisor(H)
        G_loss_S = mean_squared(H[:,1:,:], H_hat_supervise[:,:-1,:])
        GS_solver.minimize(G_loss_S, gen.trainable_variables + sup.trainable_variables)

        if itt % 1000 == 0:
            print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(G_loss_S)),4)) 
      
    print('Finish Training with Supervised Loss Only')    

    print('Start Joint Training')
  
    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            # Set mini-batch
            X_mb = batch_generator(ori_data, batch_size)               
            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, max_seq_len)

            E_hat = gen(Z_mb)
            H = emb(X_mb)
            H_hat = sup(E_hat)
            Y_fake = disc(H_hat)
            Y_fake_e = disc(E_hat)
            X_hat = rec(H_hat)
            X_tilde(rec(H))
            G_loss_U = cross_entropy(tf.ones_like(Y_fake), Y_fake)
            G_loss_U_e = cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
                
            # 2. Supervised loss
            G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
                
            # 3. Two Momments
            G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X_mb,[0])[1] + 1e-6)))
            G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X_mb,[0])[0])))
                
            G_loss_V = G_loss_V1 + G_loss_V2
            G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 

            G_solver.minimize(G_loss, gen.trainable_variables + sup.trainable_variables)
            E_loss_T0 = mean_squared(X_mb, X_tilde)
            E_loss0 = 10*tf.sqrt(E_loss_T0)
            E_loss = E_loss0  + 0.1*G_loss_S
            E_solver.minimize(E_loss, emb.trainable_variables + rec.trainable_variables)
        # Discriminator training        
        # Set mini-batch
        X_mb, = batch_generator(ori_data, batch_size)           
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, max_seq_len)
        # Check discriminator loss before updating
        H = emb(X_mb)
        E_hat = gen(Z_mb)
        Y_real = disc(H)
        H_hat = sup(E_hat)
        Y_fake = disc(H_hat)
        Y_fake_e = disc(E_hat)
        D_loss_real = cross_entropy(tf.ones_like(Y_real), Y_real)
        D_loss_fake = cross_entropy(tf.zeros_like(Y_fake), Y_fake)
        D_loss_fake_e = cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        # Train discriminator (only when the discriminator does not work well)
        if (D_loss > 0.15):        
            D_solver.minimize(D_loss, disc.trainable_variables)
            
        # Print multiple checkpoints
        if itt % 1000 == 0:
            print('step: '+ str(itt) + '/' + str(iterations) + 
                ', d_loss: ' + str(np.round(D_loss,4)) + 
                ', g_loss_u: ' + str(np.round(G_loss_U,4)) + 
                ', g_loss_s: ' + str(np.round(np.sqrt(G_loss_S),4)) + 
                ', g_loss_v: ' + str(np.round(G_loss_V,4)) + 
                ', e_loss_t0: ' + str(np.round(np.sqrt(E_loss_T0),4))  )
    print('Finish Joint Training')
    Z_mb = random_generator(no, z_dim, max_seq_len)
    E_hat = gen(Z_mb)
    H_hat = sup(E_hat)
    X_hat = rec(H_hat)
    data_curr = X_hat
    generated_data = []
    for i in range(no):
        temp = data_curr[i,:ori_time[i],:]
        generated_data.append(temp)
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data