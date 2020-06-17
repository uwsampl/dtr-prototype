z_dim = 256
g_inp = z_dim
g_hid = 128
g_out = 2

d_inp = g_out
d_hid = 128
d_out = 1

minibatch_size = 512

unrolled_steps = 10
d_learning_rate = 1e-4
g_learning_rate = 1e-3
optim_betas = (0.5, 0.999)
num_iterations = 3000
log_interval = 300
d_steps = 1
g_steps = 1

seed = 123456
use_higher = True
prefix = 'yes_higher'