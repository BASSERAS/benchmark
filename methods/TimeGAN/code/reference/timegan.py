"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import tensorflow as tf
import numpy as np

# ── TF1 compat shim for TF 2.x ───────────────────────────────────────────
# Disable eager execution so tf.placeholder() etc work
import tensorflow.compat.v1 as _tf1
_tf1.disable_eager_execution()
_tf1.disable_v2_behavior()
# Replace tf with compat.v1 so ALL tf.* calls use TF1 API
tf = _tf1
_TF1_APIS = [
    'placeholder', 'Session', 'InteractiveSession', 'reset_default_graph',
    'global_variables_initializer', 'set_random_seed', 'get_variable',
    'variable_scope', 'trainable_variables', 'losses', 'AUTO_REUSE',
    'make_template', 'get_collection', 'GraphKeys', 'get_default_graph',
    'get_default_session', 'assign',
]
for _name in _TF1_APIS:
    if not hasattr(tf, _name):
        try:
            setattr(tf, _name, getattr(_tf1, _name))
        except Exception:
            pass

# tf.train → tf.compat.v1.train
if not hasattr(tf.train, 'AdamOptimizer'):
    try:
        tf.train = _tf1.train
    except:
        pass

# tf.nn.rnn_cell and dynamic_rnn
if not hasattr(tf.nn, 'rnn_cell'):
    tf.nn.rnn_cell = _tf1.nn.rnn_cell
if not hasattr(tf.nn, 'dynamic_rnn'):
    tf.nn.dynamic_rnn = _tf1.nn.dynamic_rnn

# tf.losses — copy specific loss functions from compat v1
for _loss_name in ['sigmoid_cross_entropy', 'mean_squared_error', 'absolute_difference',
                    'log_loss', 'huber_loss', 'cosine_distance',
                    'hinge_loss', 'sparse_softmax_cross_entropy',
                    'softmax_cross_entropy']:
    if hasattr(_tf1.losses, _loss_name):
        setattr(tf.losses, _loss_name, getattr(_tf1.losses, _loss_name))
    elif hasattr(tf.losses, _loss_name):
        pass  # already exists

# tf.contrib.layers.fully_connected → tf.keras.layers.Dense
if not hasattr(tf, 'contrib'):
    tf.contrib = type('contrib', (), {})()
    tf.contrib.layers = type('layers', (), {})()
    def _fully_connected(inputs, num_outputs, activation_fn=None, **kw):
        act = activation_fn if activation_fn is not None else lambda x: x
        return tf.keras.layers.Dense(num_outputs, activation=act)(inputs)
    tf.contrib.layers.fully_connected = _fully_connected

# tf.nn.moments
if not hasattr(tf.nn, 'moments'):
    tf.nn.moments = _tf1.nn.moments

# tf.contrib.rnn → tf.keras.layers (for utils.py)
if not hasattr(tf.contrib, 'rnn'):
    tf.contrib.rnn = type('rnn', (), {})()
    def _lstm_cell(num_units, activation=None, **kw):
        act = activation if activation is not None else tf.nn.tanh
        return tf.keras.layers.LSTMCell(num_units, activation=act)
    tf.contrib.rnn.BasicLSTMCell = _lstm_cell
    try:
        tf.contrib.rnn.LayerNormBasicLSTMCell = _lstm_cell
    except:
        pass
from utils import extract_time, rnn_cell, random_generator, batch_generator


def timegan (ori_data, parameters):
  """TimeGAN function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: TimeGAN network parameters
    
  Returns:
    - generated_data: generated time-series data
  """
  # Initialization on the Graph
  tf.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
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
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
              
  ## Build a RNN networks          
  
  # Network Parameters
  hidden_dim   = parameters['hidden_dim'] 
  num_layers   = parameters['num_layer']
  iterations   = parameters['iterations']
  batch_size   = parameters['batch_size']
  module_name  = parameters['module'] 
  z_dim        = dim
  gamma        = 1
    
  # Input place holders
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "myinput_z")
  T = tf.placeholder(tf.int32, [None], name = "myinput_t")
  
  def embedder (X, T):
    """Embedding network between original feature space to latent space.
    
    Args:
      - X: input time-series features
      - T: input time information
      
    Returns:
      - H: embeddings
    """
    with tf.variable_scope("embedder", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
      H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return H
      
  def recovery (H, T):   
    """Recovery network from latent space to original space.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - X_tilde: recovered data
    """     
    with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):       
      r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
      X_tilde = tf.contrib.layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid) 
    return X_tilde
    
  def generator (Z, T):  
    """Generator function: Generate time-series data in latent space.
    
    Args:
      - Z: random variables
      - T: input time information
      
    Returns:
      - E: generated embedding
    """        
    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
      E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return E
      
  def supervisor (H, T): 
    """Generate next sequence using the previous sequence.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """          
    with tf.variable_scope("supervisor", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
      S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return S
          
  def discriminator (H, T):
    """Discriminate the original and synthetic time-series data.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """        
    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
      d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
      Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
    return Y_hat   
    
  # Embedder & Recovery
  H = embedder(X, T)
  X_tilde = recovery(H, T)
    
  # Generator
  E_hat = generator(Z, T)
  H_hat = supervisor(E_hat, T)
  H_hat_supervise = supervisor(H, T)
    
  # Synthetic data
  X_hat = recovery(H_hat, T)
    
  # Discriminator
  Y_fake = discriminator(H_hat, T)
  Y_real = discriminator(H, T)     
  Y_fake_e = discriminator(E_hat, T)
    
  # Variables        
  e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
  # Discriminator loss
  D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
  D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
  D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
  # Generator loss
  # 1. Adversarial loss
  G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
  G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
  # 2. Supervised loss
  G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
    
  # 3. Two Momments
  G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
  G_loss_V = G_loss_V1 + G_loss_V2
    
  # 4. Summation
  G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
            
  # Embedder network loss
  E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
  E_loss0 = 10*tf.sqrt(E_loss_T0)
  E_loss = E_loss0  + 0.1*G_loss_S
    
  # optimizer
  E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
  E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)      
  GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)   
        
  ## TimeGAN training
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Loss history tracking
  e_loss_history = []
  g_s_loss_history = []
  joint_d_loss_history = []
  joint_g_u_loss_history = []
  joint_g_s_loss_history = []
  joint_g_v_loss_history = []
  joint_e_loss_history = []
    
  # 1. Embedding network training
  print('Start Embedding Network Training')
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
    # Train embedder        
    _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})        
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )
    e_loss_history.append((itt, np.sqrt(float(step_e_loss))))

  print('Finish Embedding Network Training')
    
  # 2. Training only with supervised loss
  print('Start Training with Supervised Loss Only')
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)    
    # Random vector generation   
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    # Train generator       
    _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})       
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )
    g_s_loss_history.append((itt, np.sqrt(float(step_g_loss_s))))

  print('Finish Training with Supervised Loss Only')
    
  # 3. Joint Training
  print('Start Joint Training')
  
  for itt in range(iterations):
    # Generator training (twice more than discriminator training)
    for kk in range(2):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)               
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Train generator
      _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
       # Train embedder        
      _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
           
    # Discriminator training        
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
    # Random vector generation
    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    # Check discriminator loss before updating
    check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    # Train discriminator (only when the discriminator does not work well)
    if (check_d_loss > 0.15):        
      _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        
    # Print multiple checkpoints
    if itt % 1000 == 0:
      step_d_loss_val = float(step_d_loss) if 'step_d_loss' in dir() else float(check_d_loss)
      print('step: '+ str(itt) + '/' + str(iterations) +
            ', d_loss: ' + str(np.round(step_d_loss_val,4)) +
            ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) +
            ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) +
            ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) +
            ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
    joint_d_loss_history.append((itt, float(step_d_loss) if 'step_d_loss' in dir() else float(check_d_loss)))
    joint_g_u_loss_history.append((itt, float(step_g_loss_u)))
    joint_g_s_loss_history.append((itt, np.sqrt(float(step_g_loss_s))))
    joint_g_v_loss_history.append((itt, float(step_g_loss_v)))
    joint_e_loss_history.append((itt, float(step_e_loss_t0)))
  print('Finish Joint Training')
    
  ## Synthetic data generation
  Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
  generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})    
    
  generated_data = list()
    
  for i in range(no):
    temp = generated_data_curr[i,:ori_time[i],:]
    generated_data.append(temp)
        
  # Renormalization
  generated_data = generated_data * max_val
  generated_data = generated_data + min_val

  # Collect loss history for benchmark tracking
  loss_history = {
      'e_loss': e_loss_history,
      'g_s_loss': g_s_loss_history,
      'joint_d_loss': joint_d_loss_history,
      'joint_g_u_loss': joint_g_u_loss_history,
      'joint_g_s_loss': joint_g_s_loss_history,
      'joint_g_v_loss': joint_g_v_loss_history,
      'joint_e_loss': joint_e_loss_history,
  }

  return generated_data, loss_history
