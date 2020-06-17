import json
import os
import random
import torch
import copy
import numpy as np

from torch_models import word_language_model as wlm
from torch_models import treelstm
from torch_models import unet
from torch_models import lstm
from torch_models import unroll_gan
from torch_models import densenet_bc

import torch_models.pytorch_resnet_cifar10.resnet as rn

# relative because these assume they are being called from an experiment's subdirectory
WLM_DATA_PATH = '../../shared/torch_models/word_language_model/data/wikitext-2'

TREELSTM_SICK_DATA_PATH = '../../shared/torch_models/treelstm/data/sick'
TREELSTM_GLOVE_DATA_PATH = '../../shared/torch_models/treelstm/data/glove'

RESNETS = {
    '20': rn.resnet20,
    '32': rn.resnet32,
    '44': rn.resnet44,
    '56': rn.resnet56,
    '110': rn.resnet110,
    '1202': rn.resnet1202,
}

DENSENETS = {
    '100': lambda: densenet_bc.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
}

identity = lambda x: x

def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()

def get_criterion(model: str):
    if model == 'lstm':
        return torch.nn.CrossEntropyLoss().cuda()
    if model.startswith('treelstm'):
        return torch.nn.KLDivLoss().cuda()
    if model == 'unet':
        return torch.nn.BCEWithLogitsLoss().cuda()
    if model == 'unroll_gan':
        return binary_cross_entropy
    return torch.nn.CrossEntropyLoss().cuda()


def use_cudnn(model: str):
    # not using CuDNN for these: CuDNN implements the whole RNN in one function,
    # which is opaque to DTR so we cannot checkpoint within the RNN loop
    return model not in ('lstm_encoder', 'gru_encoder', 'lstm', 'treelstm')


def dispatch_by_name(model_name, stem):
    if stem not in {'resnet', 'densenet'}:
        raise Exception('Invalid model type: {}'.format(stem))

    toks = model_name.split(stem)
    model_map = RESNETS if stem == 'resnet' else DENSENETS

    if len(toks) < 2:
        raise Exception('Not a {}: {}'.format(stem, model_name))
    size = toks[1]
    if size not in model_map:
        raise Exception('Invalid {}: {}'. format(stem, model_name))
    return model_map[size]()


def prepare_vision_cnn(stem, model_name, batch_size, use_dtr=False):
    def prepare_model(extra_params=dict()):
        model = dispatch_by_name(model_name, stem)
        model.cuda()
        model.train()
        # model.zero_grad()
        if use_dtr:
            model._apply(lambda v: v.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=dict()):
        data = torch.randn(batch_size, 3, 32, 32).cuda()
        target = torch.randn(batch_size).type("torch.LongTensor").fill_(1).cuda()
        return data, target

    def run_model(criterion, model, data, target,
                    process_model=identity, process_output=identity, process_loss=identity):
        process_model(model)
        if use_dtr:
            data = data.checkpoint()
            target = target.checkpoint()
        output = model(data)
        process_output(output)
        loss = criterion(output, target)
        process_loss(loss)
        loss.backward()
        # we are not actually using the loss here
        # but a real training loop would so we have to decheckpoint
        if use_dtr:
            data = data.decheckpoint()
            loss = loss.decheckpoint()
            target = target.decheckpoint()

        # we include these deletions for generating logs;
        # these will ensure these fields are deallocated
        # before the end of the log, so anything still live
        # will be a gradient or weight
        del data
        del loss
        del target

    # note: we anticipated having to do untimed teardown
    # but did not end up doing any, hence these empty functions
    def teardown(model):
        pass

    return prepare_model, gen_input, run_model, teardown


def prepare_word_language_model(model_name, batch_size, use_dtr=False):
    corpus = wlm.data.Corpus(WLM_DATA_PATH)
    ntokens = len(corpus.dictionary)
    train_data = wlm.main.batchify(corpus.train, batch_size)

    if model_name in {'lstm_encoder', 'gru_encoder'}:
        def create_model(extra_params=dict()):
            cell_name = 'LSTM' if model_name == 'lstm_encoder' else 'GRU'
            model = wlm.model.RNNModel(cell_name, ntokens, 200, 200, 2, 0.2, False)
            model.train()
            model.zero_grad()
            model.cuda()
            if use_dtr:
                model._apply(lambda v: v.detach().checkpoint())
            hidden = model.init_hidden(batch_size)

            # hidden state for LSTM is a tuple, GRU is a tensor
            if use_dtr:
                if model_name == 'lstm_encoder':
                    new_hidden = []
                    for t in hidden:
                        new_hidden.append(t.checkpoint())
                    hidden = tuple(new_hidden)
                else:
                    hidden = hidden.checkpoint()

            return [model, hidden]

        def gen_input(i, extra_params=dict()):
            # TODO should probably pick a random batch
            data, targets = wlm.main.get_batch(train_data, random.randint(i, len(train_data)) - 1)
            data = data.cuda()
            targets = targets.cuda()
            return data, targets

        def run_model(criterion, model, hidden, data, targets,
                            process_model=identity, process_output=identity, process_loss=identity):
            process_model(model)
            if use_dtr:
                data = data.checkpoint()
                targets = targets.checkpoint()

            output, new_hidden = model(data, hidden)
            process_output(output)
            loss = criterion(output.view(-1, ntokens), targets)
            process_loss(loss)
            loss.backward()
            wlm.main.repackage_hidden(new_hidden)
            # we are not actually using the loss here
            # but a real training loop would so we have to decheckpoint
            if use_dtr:
                loss = loss.decheckpoint()

        def teardown(model, hidden):
            pass

        return create_model, gen_input, run_model, teardown

    raise Exception('Invalid model {}'.format(model_name))


def setup_treelstm_data():
    train_dir = os.path.join(TREELSTM_SICK_DATA_PATH, 'train/')
    dev_dir = os.path.join(TREELSTM_SICK_DATA_PATH, 'dev/')
    test_dir = os.path.join(TREELSTM_SICK_DATA_PATH, 'test/')

    sick_vocab_file = os.path.join(TREELSTM_SICK_DATA_PATH, 'sick.vocab')
    if not os.path.isfile(sick_vocab_file):
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files = token_files_a + token_files_b
        sick_vocab_file = os.path.join(TREELSTM_SICK_DATA_PATH, 'sick.vocab')
        treelstm.utils.build_vocab(token_files, sick_vocab_file)

    vocab = treelstm.vocab.Vocab(filename=sick_vocab_file,
                  data=[treelstm.Constants.PAD_WORD, treelstm.Constants.UNK_WORD,
                        treelstm.Constants.BOS_WORD, treelstm.Constants.EOS_WORD])

    train_file = os.path.join(TREELSTM_SICK_DATA_PATH, 'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = treelstm.dataset.SICKDataset(train_dir, vocab, 5)
        torch.save(train_dataset, train_file)

    dataset = train_dataset

    emb_file = os.path.join(TREELSTM_SICK_DATA_PATH, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = treelstm.utils.load_word_vectors(
            os.path.join(TREELSTM_GLOVE_DATA_PATH, 'glove.840B.300d'))
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float).cuda()
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([treelstm.Constants.PAD_WORD, treelstm.Constants.UNK_WORD,
                                    treelstm.Constants.BOS_WORD, treelstm.Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)

    return vocab, dataset, emb

# treelstm from treelstm.pytorch
def prepare_treelstm_old(batch_size, use_dtr=False):
    vocab, dataset, emb = setup_treelstm_data()

    def create_model(extra_params=dict()):
        model = treelstm.model.SimilarityTreeLSTM(
            vocab.size(),
            300, # input_dim
            150, # mem_dim
            50,  # hidden_dim
            5,   # num_classes
            False,
            False)

        model.cuda()
        emb_ = emb.cuda()
        model.emb.weight.data.copy_(emb_)
        model.cuda()
        model.train()

        if use_dtr:
            model._apply(lambda t: t.detach().checkpoint())
        return [model]

    def gen_input(i, extra_params=dict()):
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        ltree, linput, rtree, rinput, label = dataset[indices[i]]
        target = treelstm.utils.map_label_to_target(label, dataset.num_classes)
        linput = linput.cuda()
        rinput = rinput.cuda()
        target = target.cuda()
        return ltree, linput, rtree, rinput, target

    def run_model(criterion, model, ltree, linput, rtree, rinput, target,
                        process_model=identity, process_output=identity, process_loss=identity):
        process_model(model)
        if use_dtr:
            linput = linput.checkpoint()
            rinput = rinput.checkpoint()
            target = target.checkpoint()
        output = model(ltree, linput, rtree, rinput)
        process_output(output)
        loss = criterion(output, target)
        process_loss(loss)
        loss.backward()
        if use_dtr:
            loss.decheckpoint()
            linput.decheckpoint()
            rinput = rinput.decheckpoint()
            target = target.decheckpoint()
        
        del linput
        del rinput
        del target
        del loss

    def teardown(model):
        pass

    return create_model, gen_input, run_model, teardown

# our implementation of treelstm
def prepare_treelstm(batch_size, use_dtr=False):
    in_dim = 100
    mem_dim = 300

    # hidden and memory dims cranked up to increase memory use
    def create_model(extra_params=dict()):
        model = treelstm.model.TreeLSTM(extra_params.get('in_dim', in_dim), extra_params.get('mem_dim', mem_dim), use_dtr)

        model.cuda()
        model.train()

        if use_dtr:
            model._apply(lambda t: t.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=dict()):
        ltree = treelstm.utils.gen_tree(extra_params.get('in_dim', in_dim), batch_size,
                                        depth=extra_params.get('dep', 5), use_dtr=use_dtr)
        return [ltree]

    def run_model(criterion, model, ltree,
                        process_model=identity, process_output=identity, process_loss=identity):
        if use_dtr:
            ltree.map_(lambda x: x.detach().checkpoint())

        output = model(ltree)
        output = torch.sum(output)
        output.backward()

        if use_dtr:
            output = output.decheckpoint()
            ltree.map_(lambda x: x.decheckpoint())
        
        del output
        del ltree

    def teardown(model):
        pass

    return create_model, gen_input, run_model, teardown


def prepare_unet(batch_size, use_dtr):
    n_channels=3
    def prepare_model(extra_params=dict()):
        # settings taken from repo
        model = unet.UNet(n_channels=n_channels, n_classes=1, bilinear=True)
        model.cuda()
        model.train()
        model.zero_grad()
        if use_dtr:
            model._apply(lambda v: v.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=dict()):
        data = torch.randn(batch_size, n_channels, 512, 512).cuda()
        target = torch.randn(batch_size, 1, 512, 512).cuda()
        return data, target

    def run_model(criterion, model, data, target,
                    process_model=identity, process_output=identity, process_loss=identity):
        process_model(model)
        if use_dtr:
            data = data.checkpoint()
            target = target.checkpoint()
        output = model(data)
        process_output(output)
        loss = criterion(output, target)
        process_loss(loss)
        loss.backward()
        # we are not actually using the loss here
        # but a real training loop would so we have to decheckpoint
        if use_dtr:
            data = data.decheckpoint()
            target = target.decheckpoint()
            loss = loss.decheckpoint()

        del data
        del target
        del loss
        del output

    def teardown(model):
        pass

    return prepare_model, gen_input, run_model, teardown

def prepare_lstm_classification(batch_size, use_dtr):
    # Default values:
    # embedding_dim = 100
    # hidden_dim = 50
    # sentence_len = 32

    def prepare_model(extra_params=dict()):
        model = lstm.LSTM(extra_params.get('in_dim', 10), extra_params.get('mem_dim', 300), use_dtr)
        model.cuda()
        model.train()

        if use_dtr:
            model._apply(lambda v: v.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=dict()):
        data = [torch.zeros(extra_params.get('input_size', 50), extra_params.get('in_dim', 50)).cuda()
                for _ in range(batch_size)]
        return [data]

    def run_model(criterion, model, data,
                    process_model=identity, process_output=identity, process_loss=identity):
        # process_model(model)
        # target = torch.squeeze(target)
        if use_dtr:
            data = list(map(lambda x: x.checkpoint(), data))

        output = model(data)
        output = torch.sum(output[-1])
        output.backward()
        # we are not actually using the loss here
        # but a real training loop would so we have to decheckpoint
        if use_dtr:
            data = list(map(lambda x: x.decheckpoint(), data))

        del output
        del data

    def teardown(model):
        pass

    return prepare_model, gen_input, run_model, teardown

# Unrolled GAN implementation adapted from https://github.com/mk-minchul/unroll_gan

def prepare_unrolled_gan(batch_size, use_dtr):
    configs = unroll_gan.configs.yes_higher_unroll
    import higher

    def prepare_model(extra_params=dict()):
        G = unroll_gan.Generator(input_size=configs.g_inp, hidden_size=configs.g_hid, output_size=configs.g_out)
        D = unroll_gan.Discriminator(input_size=configs.d_inp, hidden_size=configs.d_hid, output_size=configs.d_out)
        G.zero_grad()
        G.train()
        G.cuda()

        D.zero_grad()
        D.train()
        D.cuda()

        if use_dtr:
            G._apply(lambda t: t.detach().checkpoint())
            D._apply(lambda t: t.detach().checkpoint())

        d_optim, g_optim = [torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999)),
                            torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))]

        return [D, G, d_optim, g_optim]

    def gen_input(i, extra_params=dict()):
        dset = unroll_gan.gaussian_data_generator(configs.seed)
        dset.random_distribution()
        return [dset]
    
    def d_loop(dset, G, D, d_optimizer, criterion):
        # 1. Train D on real+fake
        d_optimizer.zero_grad()

        #  1A: Train D on real
        d_real_data = torch.from_numpy(dset.sample(configs.minibatch_size)).cuda()
        d_real_decision = D(d_real_data)
        target = torch.ones_like(d_real_decision).cuda()
        d_real_error = criterion(d_real_decision, target)  # ones = true

        #  1B: Train D on fake
        d_gen_input = torch.from_numpy(unroll_gan.noise_sampler(configs.minibatch_size, configs.g_inp)).cuda()

        with torch.no_grad():
            d_fake_data = G(d_gen_input)
        d_fake_decision = D(d_fake_data)
        target = torch.zeros_like(d_fake_decision).cuda()
        d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

        d_loss = d_real_error + d_fake_error
        d_loss.backward()
        d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        return d_real_error.cpu().item(), d_fake_error.cpu().item()
    
    def d_unrolled_loop_higher(dset, G, D, d_optimizer, criterion, d_gen_input=None):
        #  1A: Train D on real
        d_real_data = torch.from_numpy(dset.sample(configs.minibatch_size)).cuda()
        d_real_decision = D(d_real_data)
        target = torch.ones_like(d_real_decision).cuda()
        d_real_error = criterion(d_real_decision, target)  # ones = true

        #  1B: Train D on fake
        if d_gen_input is None:
            d_gen_input = torch.from_numpy(unroll_gan.noise_sampler(configs.minibatch_size, configs.g_inp)).cuda()

        d_fake_data = G(d_gen_input)
        d_fake_decision = D(d_fake_data)
        target = torch.zeros_like(d_fake_decision).cuda()
        d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

        d_loss = d_real_error + d_fake_error
        d_optimizer.step(d_loss)  # note that `step` must take `loss` as an argument!

        return d_real_error.cpu().item(), d_fake_error.cpu().item()
    
    def d_unrolled_loop(dset, G, D, d_optimizer, criterion, d_gen_input=None):
        # 1. Train D on real+fake
        d_optimizer.zero_grad()

        #  1A: Train D on real
        d_real_data = torch.from_numpy(dset.sample(configs.minibatch_size)).cuda()
        d_real_decision = D(d_real_data)
        target = torch.ones_like(d_real_decision).cuda()
        d_real_error = criterion(d_real_decision, target)  # ones = true

        #  1B: Train D on fake
        if d_gen_input is None:
            d_gen_input = torch.from_numpy(unroll_gan.noise_sampler(configs.minibatch_size, configs.g_inp)).cuda()

        with torch.no_grad():
            d_fake_data = G(d_gen_input)
        # d_fake_data = G(d_gen_input)
        d_fake_decision = D(d_fake_data)
        target = torch.zeros_like(d_fake_decision).cuda()
        d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

        d_loss = d_real_error + d_fake_error
        d_loss.backward(create_graph=True)
        d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
        return d_real_error.cpu().item(), d_fake_error.cpu().item()

    def g_loop(dset, G, D, g_optimizer, d_optimizer, criterion):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        gen_input = torch.from_numpy(unroll_gan.noise_sampler(configs.minibatch_size, configs.g_inp)).cuda()

        if configs.unrolled_steps > 0:
            if configs.use_higher:
                backup = copy.deepcopy(D)

                with higher.innerloop_ctx(D, d_optimizer) as (functional_D, diff_D_optimizer):
                    for i in range(configs.unrolled_steps):
                        d_unrolled_loop_higher(dset, G, functional_D, diff_D_optimizer, criterion, d_gen_input=None)

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
                for i in range(configs.unrolled_steps):
                    d_unrolled_loop(dset, G, D, d_optimizer, criterion, d_gen_input=gen_input)

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

        return g_error.cpu().item()

    def run_model(criterion, D, G, d_optim, g_optim, dset,
                    process_model=identity, process_output=identity, process_loss=identity):
        samples = []
        d_infos = []

        for d_index in range(configs.d_steps):
            d_info = d_loop(dset, G, D, d_optim, criterion)
            d_infos.append(d_info)
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos

        g_infos = []
        for g_index in range(configs.g_steps):
            g_info = g_loop(dset, G, D, g_optim, d_optim, criterion)
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos

        del dset

    def teardown(D, G, d_optim, g_optim):
        pass

    return prepare_model, gen_input, run_model, teardown

def prepare_model(model_name, batch_size, use_dtr=False):
    if model_name.startswith('resnet'):
        return prepare_vision_cnn('resnet', model_name, batch_size, use_dtr)

    if model_name.startswith('densenet'):
        return prepare_vision_cnn('densenet', model_name, batch_size, use_dtr)

    if model_name in {'lstm_encoder', 'gru_encoder'}:
        return prepare_word_language_model(model_name, batch_size, use_dtr)

    # our rewritten treelstm
    if model_name == 'treelstm':
        return prepare_treelstm(batch_size, use_dtr)

    # the original treelstm.pytorch
    if model_name == 'treelstm_old':
        return prepare_treelstm_old(batch_size, use_dtr)

    if model_name == 'unet':
        return prepare_unet(batch_size, use_dtr)

    # our rewritten LSTM
    if model_name == 'lstm':
        return prepare_lstm_classification(batch_size, use_dtr)

    if model_name == 'unroll_gan':
        return prepare_unrolled_gan(batch_size, use_dtr)

    raise Exception('Model {} not supported'.format(model_name))
