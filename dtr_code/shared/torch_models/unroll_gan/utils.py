import torch
import higher
import copy
# from .configs import yes_higher_unroll as config
from .data import noise_sampler

def d_loop(config, dset, G, D, d_optimizer, criterion):
        # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision).cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    d_gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision).cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

    # return d_real_error.item(), d_fake_error.item()

def d_unrolled_loop_higher(config, dset, G, D, d_optimizer, criterion, d_gen_input=None):
    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision).cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()

    d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision).cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_optimizer.step(d_loss)  # note that `step` must take `loss` as an argument!

    # return d_real_error().item(), d_fake_error.item()

def d_unrolled_loop(config, dset, G, D, d_optimizer, criterion, d_gen_input=None):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(config.minibatch_size)).cuda()
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision).cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    # d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision).cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward(create_graph=True)
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    # return d_real_error, d_fake_error

def g_loop(config, dset, G, D, g_optimizer, d_optimizer, criterion):
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(config.minibatch_size, config.g_inp)).cuda()

    if config.unrolled_steps > 0:
        if config.use_higher:
            backup = copy.deepcopy(D)

            with higher.innerloop_ctx(D, d_optimizer) as (functional_D, diff_D_optimizer):
                for i in range(config.unrolled_steps):
                    d_unrolled_loop_higher(config, dset, G, functional_D, diff_D_optimizer, criterion, d_gen_input=None)

                g_optimizer.zero_grad()
                g_fake_data = G(gen_input)
                dg_fake_decision = functional_D(g_fake_data)
                target = torch.ones_like(dg_fake_decision).cuda()
                g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters

            D.load(backup)
            del backup
        else:
            backup = copy.deepcopy(D)
            for i in range(config.unrolled_steps):
                d_unrolled_loop(config, dset, G, D, d_optimizer, criterion, d_gen_input=gen_input)

            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data)
            target = torch.ones_like(dg_fake_decision).cuda()
            g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters
            D.load(backup)
            del backup

    else:
        g_fake_data = G(gen_input)
        dg_fake_decision = D(g_fake_data)
        target = torch.ones_like(dg_fake_decision).cuda()
        g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    # return g_error.item()
