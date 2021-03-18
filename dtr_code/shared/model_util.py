import json
import os
import random
import torch
import copy
import numpy as np

from torch_models import word_language_model as wlm
from torch_models import vision_models as vm
from torch_models import treelstm
from torch_models import unet
from torch_models import lstm
from torch_models import unroll_gan
from torch_models import densenet_bc
from torch_models import inceptionv4
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch_models.pytorch_resnet_cifar10.resnet as rn

# relative because these assume they are being called from an experiment's subdirectory
WLM_DATA_PATH = '../../shared/torch_models/word_language_model/data/wikitext-2'

TREELSTM_SICK_DATA_PATH = '../../shared/torch_models/treelstm/data/sick'
TREELSTM_GLOVE_DATA_PATH = '../../shared/torch_models/treelstm/data/glove'

LSTM_DATA_PATH = '../../shared/torch_models/lstm/data'

TV_RESNETS = {
    # Torchvision resnets
    '18': vm.resnet18,
    '34': vm.resnet34,
    '50': vm.resnet50,
    '101': vm.resnet101,
    '152': vm.resnet152
}

CIFAR_RESNETS = {
    # CIFAR resnets
    '20': rn.resnet20,
    '32': rn.resnet32,
    '44': rn.resnet44,
    '56': rn.resnet56,
    '110': rn.resnet110,
    '1202': rn.resnet1202,
}

TV_DENSENETS = {
    # Torchvision densenets
    '121': vm.densenet121,
    '161': vm.densenet161,
    '169': vm.densenet169,
    '201': vm.densenet201
}

DENSENET_BC = {
    '100': lambda: densenet_bc.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
}
identity = lambda x: x

def get_optimizer(model_name : str, models):
    model = models[0]
    if model_name == 'unroll_gan':
        return None
    if model_name == 'unet':
        return torch.optim.RMSprop(model.parameters(), 0.001, weight_decay=1e-8, momentum=0.9)
    return torch.optim.SGD(model.parameters(), 0.1, 0.9)


def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()


def format_model_name(model_name, specific_params):
    """
    Given the model name and input parameters,
    return a string ready to include as a name field in simulated graphs.
    """
    batch_size = specific_params['batch_size']
    if 'resnet' in model_name:
        layers = ''.join(filter(lambda x: x.isdigit(), model_name))
        return f'ResNet-{layers} ({batch_size})'
    if 'densenet' in model_name:
        layers = ''.join(filter(lambda x: x.isdigit(), model_name))
        return f'DenseNet-{layers} ({batch_size})'
    if 'inception' in model_name:
        version = model_name[-1]
        return f'Inception V{version} ({batch_size})'
    if 'treelstm' in model_name:
        return 'TreeLSTM'
    if model_name == 'unroll_gan':
        return 'Unrolled GAN'
    if model_name == 'lstm':
        return f'LSTM ({batch_size})'
    return model_name


def format_input_description(model_name, specific_params):
    """
    Given the model name and input parameters,
    return a string ready to include as an input description in simulated graphs.
    """
    if 'treelstm' in model_name:
        depth = specific_params['batch_size']
        height = specific_params.get('input_size', 100)
        width = specific_params.get('in_dim', 300)
        return f'Binary tree of depth {depth}, node size {height}x{width}'
    if 'lstm' in model_name:
        length = specific_params['batch_size']
        hidden_dim = specific_params.get('mem_dim', 300)
        input_dim = specific_params.get('in_dim', 10)
        return f'Input dimension {input_dim},\nHidden dimension {hidden_dim},\nSequence length {length}'
    if model_name == 'transformer':
        input_seq = specific_params.get('input_seq_length', 10)
        target_seq = specific_params.get('target_seq_length', 20)
        feature = specific_params.get('num_feature', 512)
        return f'Input seq {input_seq},\ntarget seq {target_seq},\n{feature} features'
    if model_name == 'unroll_gan':
        configs = unroll_gan.configs.yes_higher_unroll
        steps = specific_params.get('unrolled_steps', configs.unrolled_steps)
        d_hid = specific_params.get('d_hid', configs.d_hid)
        g_hid = specific_params.get('g_hid', configs.g_hid)
        return f'{steps} steps, {d_hid}x{g_hid}'
    height, width = 32, 32
    if 'inception' in model_name:
        height, width = 299, 299
    if 'unet' in model_name:
        height, width = 416, 608
    height, width = specific_params.get('height', height), specific_params.get('width', width)
    return f'{height}x{width}'


def get_criterion(model: str):
    if model == 'lstm':
        return torch.nn.CrossEntropyLoss().cuda()
    if model.startswith('treelstm'):
        return torch.nn.KLDivLoss().cuda()
    if model == 'unet':
        return torch.nn.BCEWithLogitsLoss().cuda()
    if model == 'unroll_gan':
        return binary_cross_entropy
    if model == 'transformer':
        return torch.nn.NLLLoss().cuda()
    return torch.nn.CrossEntropyLoss().cuda()


def get_optimizer(model_name : str, models):
    # return None
    model = models[0]
    if model_name == 'unroll_gan':
                return [torch.optim.Adam(models[0].parameters(), lr=1e-4, betas=(0.5, 0.999)),
                        torch.optim.Adam(models[1].parameters(), lr=1e-3, betas=(0.5, 0.999))]
    if model_name == 'unroll_gan':
        return [torch.optim.Adam(models[0].parameters(), lr=1e-4, betas=(0.5, 0.999)),
                torch.optim.Adam(models[1].parameters(), lr=1e-3, betas=(0.5, 0.999))]
    if model_name == 'unroll_gan':
        return [torch.optim.Adam(models[0].parameters(), lr=1e-4, betas=(0.5, 0.999)),
                torch.optim.Adam(models[1].parameters(), lr=1e-3, betas=(0.5, 0.999))]
    if model_name == 'unet':
        return torch.optim.RMSprop(model.parameters(), 0.001, weight_decay=1e-8, momentum=0.9)
    return torch.optim.SGD(model.parameters(), 0.1, 0.9)


def use_cudnn(model: str):
    # not using CuDNN for these: CuDNN implements the whole RNN in one function,
    # which is opaque to DTR so we cannot checkpoint within the RNN loop
    return model not in ('lstm_encoder', 'gru_encoder')


def dispatch_by_name(model_name, stem):
    if stem not in {'resnet', 'densenet', 'tv_resnet', 'tv_densenet'}:
        raise Exception('Invalid model type: {}'.format(stem))

    toks = model_name.split(stem)
    model_map = TV_RESNETS if stem == 'tv_resnet'\
                else CIFAR_RESNETS if stem == 'resnet'\
                else TV_DENSENETS  if stem == 'tv_densenet'\
                else DENSENET_BC

    if len(toks) < 2:
        raise Exception('Not a {}: {}'.format(stem, model_name))
    size = toks[1]
    if size not in model_map:
        raise Exception('Invalid {}: {}'. format(stem, model_name))
    return model_map[size]()


def prepare_vision_cnn(stem, model_name, batch_size, use_dtr=False):
    def prepare_model(extra_params=None):
        model = dispatch_by_name(model_name, stem)
        model.cuda()
        model.train()
        # model.zero_grad()
        if use_dtr:
            model._apply(lambda v: v.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=None):
        height, width = (extra_params.get('height', 32), extra_params.get('width', 32)) if extra_params is not None else (32, 32)
        data = torch.randn(batch_size, 3, height, width).cuda()
        target = torch.randn(batch_size).type("torch.LongTensor").fill_(1).cuda()
        return data, target

    def run_model(criterion, model, data, target,
                  process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
        process_model(model)
        if use_dtr:
            data = data.checkpoint()
            target = target.checkpoint()
        output = model(data)
        process_output(output)
        loss = criterion(output, target)
        process_loss(loss)
        if optimizer:
            optimizer.zero_grad()
        if use_dtr:
            torch.annotate_log('BACKWARD')
        loss.backward()
        # we are not actually using the loss here
        # but a real training loop would so we have to decheckpoint
        if use_dtr:
            data = data.decheckpoint()
            loss = loss.decheckpoint()
            target = target.decheckpoint()

        if optimizer:
            optimizer.step()

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

    if model_name == 'transformer_encoder':
        # based on main.py in the word_language_model example, with
        # default values taken from the provided arguments
        def create_model(extra_params=None):
            model = wlm.model.TransformerModel(ntokens, 200, 2, 200, 2, 0.2)
            model.train()
            model.zero_grad()
            model.cuda()
            if use_dtr:
                model._apply(lambda v: v.detach().checkpoint())
            return [model]

        def gen_input(i, extra_params=None):
            # TODO should probably pick a random batch
            data, targets = wlm.main.get_batch(train_data, i)
            return data.cuda(), targets.cuda()

        def run_model(criterion, model, data, targets,
                      process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
            process_model(model)
            if use_dtr:
                data = data.checkpoint()
                targets = targets.checkpoint()
            output = model(data)
            process_output(output)
            loss = criterion(output.view(-1, ntokens), targets)
            process_loss(loss)
            if use_dtr:
                torch.annotate_log('BACKWARD')
            loss.backward()

            del data
            del targets
            del loss

        def teardown(model, optimizer=None):
            model.zero_grad()

        return create_model, gen_input, run_model, teardown

    if model_name in {'lstm_encoder', 'gru_encoder'}:
        def create_model(extra_params=None):
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

        def gen_input(i, extra_params=None):
            # TODO should probably pick a random batch
            data, targets = wlm.main.get_batch(train_data, random.randint(i, len(train_data)) - 1)
            data = data.cuda()
            targets = targets.cuda()
            return data, targets

        def run_model(criterion, model, hidden, data, targets,
                            process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
            process_model(model)
            if use_dtr:
                data = data.checkpoint()
                targets = targets.checkpoint()

            output, new_hidden = model(data, hidden)
            process_output(output)
            loss = criterion(output.view(-1, ntokens), targets)
            process_loss(loss)
            if use_dtr:
                torch.annotate_log('BACKWARD')
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

    def create_model(extra_params=None):
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

    def gen_input(i, extra_params=None):
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        ltree, linput, rtree, rinput, label = dataset[indices[i]]
        target = treelstm.utils.map_label_to_target(label, dataset.num_classes)
        linput = linput.cuda()
        rinput = rinput.cuda()
        target = target.cuda()
        return ltree, linput, rtree, rinput, target

    def run_model(criterion, model, ltree, linput, rtree, rinput, target,
                  process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
        process_model(model)
        if use_dtr:
            linput = linput.checkpoint()
            rinput = rinput.checkpoint()
            target = target.checkpoint()
        output = model(ltree, linput, rtree, rinput)
        process_output(output)
        loss = criterion(output, target)
        process_loss(loss)
        if use_dtr:
            torch.annotate_log('BACKWARD')
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
    depth = 5

    # hidden and memory dims cranked up to increase memory use
    def create_model(extra_params=None):
        in_dim, mem_dim = ((in_dem, mem_dim) if extra_params is None
                           else (extra_params['in_dim'], extra_params['mem_dim']))
        model = treelstm.model.TreeLSTM(in_dim, mem_dim, use_dtr)

        model.cuda()
        model.train()

        if use_dtr:
            model._apply(lambda t: t.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=None):
        in_dim, depth = ((in_dim, depth) if extra_params is None
                         else (extra_params['in_dim'], extra_params.get('dep', 6)))
        ltree = treelstm.utils.gen_tree(in_dim, extra_params.get('input_size', 100),
                                        depth=batch_size, use_dtr=use_dtr)
        return [ltree]

    def run_model(criterion, model, ltree,
                  process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
        if use_dtr:
            ltree.map_(lambda x: x.detach().checkpoint())

        output = model(ltree)
        output = torch.sum(output)
        if use_dtr:
            torch.annotate_log('BACKWARD')
        output.backward()

        if use_dtr:
            output = output.decheckpoint()
            ltree.map_(lambda x: x.decheckpoint())

        del output
        del ltree

    def teardown(model):
        pass

    return create_model, gen_input, run_model, teardown


def prepare_inception(model_name, batch_size, use_dtr):
    parse_version = model_name.split('v')
    assert len(parse_version) == 2
    version = parse_version[1]

    def prepare_model(extra_params=None):
        # default settings in their respective sources
        if version == '3':
            # init weights set to false because, per the FutureWarning,
            # the initialization is really slow and we are not really training the model
            model = vm.Inception3(num_classes=1000, init_weights=False)
        else:
            model = inceptionv4.InceptionV4(num_classes=1001)
        model.cuda()
        model.train()
        model.zero_grad()
        if use_dtr:
            model._apply(lambda v: v.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=None):
        # the sources are very explicit that they expect input that is batch_size x 3 x 299 x 299
        data = torch.randn(batch_size, 3, 299, 299).cuda()
        target = torch.randn(batch_size).type("torch.LongTensor").fill_(1).cuda()
        return data, target

    def run_model(criterion, model, data, target,
                  process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
        process_model(model)
        if use_dtr:
            data = data.checkpoint()
            target = target.checkpoint()
        output = model(data)
        # Torchvision's inception v3 has two outputs
        if version == '3':
            output = output[0]
        process_output(output)
        loss = criterion(output, target)
        process_loss(loss)
        if optimizer:
            optimizer.zero_grad()
        if use_dtr:
            torch.annotate_log('BACKWARD')
        loss.backward()
        # we are not actually using the loss here
        # but a real training loop would so we have to decheckpoint
        if use_dtr:
            data = data.decheckpoint()
            loss = loss.decheckpoint()
            target = target.decheckpoint()

        if optimizer:
            optimizer.step()

        del data
        del loss
        del target

    def teardown(model):
        pass

    return prepare_model, gen_input, run_model, teardown


def prepare_transformer(batch_size, use_dtr):
    def prepare_model(extra_params=None):
        # TODO: these parameters came from the PyTorch docs
        # example but we should make sure they are settings that make sense
        model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
        model.cuda()
        model.train()
        model.zero_grad()
        if use_dtr:
            model._apply(lambda v: v.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=None):
        # TODO: parameters came from the PT example,
        # we should make these configurable
        # and ensure they correspond to realistic sizes

        # for example, it seems perplexing that the second dimension
        # is called the "batch size",
        # yet it is not needed for the criterion
        src = torch.rand((extra_params.get('input_seq_length', 10), batch_size, extra_params.get('num_feature', 512))).cuda()
        tgt = torch.rand((extra_params.get('target_seq_length', 20), batch_size, extra_params.get('num_feature', 512))).cuda()
        crit_target = torch.rand(extra_params.get('target_seq_length', 20), extra_params.get('num_feature', 512)).type('torch.LongTensor').fill_(1).cuda()
        return src, tgt, crit_target

    def run_model(criterion, model, data, target, crit_target,
                  process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
        process_model(model)
        if use_dtr:
            data = data.checkpoint()
            target = target.checkpoint()
            crit_target = crit_target.checkpoint()

        output = model(data, target)
        process_output(output)
        loss = criterion(output, crit_target)
        process_loss(loss)

        if optimizer:
            optimizer.zero_grad()
        if use_dtr:
            torch.annotate_log('BACKWARD')

        loss.backward()
        # we are not actually using the loss here
        # but a real training loop would so we have to decheckpoint
        if use_dtr:
            data = data.decheckpoint()
            target = target.decheckpoint()
            crit_target = crit_target.decheckpoint()
            loss = loss.decheckpoint()

        if optimizer:
            optimizer.step()

        del data
        del loss
        del target
        del crit_target

    def teardown(model):
        pass

    return prepare_model, gen_input, run_model, teardown


def prepare_unet(batch_size, use_dtr):
    n_channels=3
    def prepare_model(extra_params=None):
        # settings taken from repo
        model = unet.UNet(n_channels=n_channels, n_classes=1, bilinear=True)
        model.cuda()
        model.train()
        model.zero_grad()
        if use_dtr:
            model._apply(lambda v: v.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=None):
        height, width = ((416, 608) if extra_params is None else (extra_params['height'], extra_params['width']))
        data = torch.randn(batch_size, n_channels, height, width).cuda()
        target = torch.randn(batch_size, 1, height, width).cuda()
        return data, target

    def run_model(criterion, model, data, target,
                  process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
        process_model(model)
        if use_dtr:
            data = data.checkpoint()
            target = target.checkpoint()
        output = model(data)
        process_output(output)
        loss = criterion(output, target)
        process_loss(loss)
        if optimizer:
            optimizer.zero_grad()
        if use_dtr:
            torch.annotate_log('BACKWARD')
        loss.backward()
        # we are not actually using the loss here
        # but a real training loop would so we have to decheckpoint
        if use_dtr:
            data = data.decheckpoint()
            target = target.decheckpoint()
            loss = loss.decheckpoint()

        if optimizer:
            optimizer.step()

        del data
        del target
        del loss
        del output

        # TODO: uncomment the following line after graident clipping implemented
        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        # optimizer.step()

    def teardown(model, optimizer=None):
        # model.zero_grad()
        if optimizer is not None:
            optimizer.zero_grad()
        if use_dtr:
            pass
            # torch.clear_checkpointpool()

    return prepare_model, gen_input, run_model, teardown


def prepare_lstm_classification(batch_size, use_dtr):
    # Default values:
    # embedding_dim = 100
    # hidden_dim = 50
    # sentence_len = 32

    def prepare_model(extra_params=None):
        in_dim, mem_dim = (extra_params.get('in_dim', 10), extra_params.get('mem_dim', 300))
        model = lstm.LSTM(in_dim, mem_dim, use_dtr)
        model.cuda()
        model.train()

        if use_dtr:
            model._apply(lambda v: v.detach().checkpoint())

        return [model]

    def gen_input(i, extra_params=None):
        input_size, in_dim = (extra_params.get('input_size', 50), extra_params.get('in_dim', 10))
        data = [torch.zeros(input_size, in_dim).cuda()
                for _ in range(batch_size)]
        return [data]

    def run_model(criterion, model, data,
                  process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
        # process_model(model)
        # target = torch.squeeze(target)
        if use_dtr:
            data = list(map(lambda x: x.checkpoint(), data))

        output = model(data)
        output = torch.sum(output[-1])
        if use_dtr:
            torch.annotate_log('BACKWARD')
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
        configs.minibatch_size = batch_size
        configs.z_dim = extra_params.get('g_inp', configs.g_inp)
        configs.g_inp = extra_params.get('g_inp', configs.z_dim)
        configs.g_out = extra_params.get('g_out', configs.g_out)
        configs.g_hid = extra_params.get('g_hid', configs.g_hid)
        configs.d_inp = configs.g_out
        configs.unrolled_steps = extra_params.get('unrolled_steps', configs.unrolled_steps)
        configs.d_hid = extra_params.get('d_hid', configs.d_hid)
        configs.d_out = extra_params.get('d_out', configs.d_out)
        G = unroll_gan.Generator(input_size=configs.g_inp,
                                hidden_size=configs.g_hid,
                                output_size=configs.g_out)
        D = unroll_gan.Discriminator(input_size=configs.d_inp,
                                    hidden_size=configs.d_hid,
                                    output_size=configs.d_out)
        G.zero_grad()
        G.train()
        G.cuda()

        D.zero_grad()
        D.train()
        D.cuda()

        if use_dtr:
            G._apply(lambda t: t.detach().checkpoint())
            D._apply(lambda t: t.detach().checkpoint())
        return [D, G]

    def gen_input(i, extra_params=dict()):
        dset = unroll_gan.gaussian_data_generator(configs.seed, n=batch_size,
                                        std=extra_params.get('std', 0.02),
                                        radius=extra_params.get('radius', 2))
        dset.random_distribution()
        return [dset]

    def run_model(criterion, D, G, dset,
                    process_model=identity, process_output=identity, process_loss=identity, optimizer=None):
        samples = []
        d_infos = []
        d_optim, g_optim = get_optimizer('unroll_gan', [D, G]) 
        for d_index in range(configs.d_steps):
            unroll_gan.d_loop(configs, dset, G, D, d_optim, criterion)
            # d_infos.append(d_info)
        # d_infos = np.mean(d_infos, 0)
        # d_real_loss, d_fake_loss = d_infos

        g_infos = []
        for g_index in range(configs.g_steps):
            unroll_gan.g_loop(configs, dset, G, D, g_optim, d_optim, criterion)
            # g_infos.append(g_info)
        # g_infos = np.mean(g_infos)
        # g_loss = g_infos

        del dset

    def teardown(D, G, optimizer=None):
        if use_dtr:
            pass
            # torch.clear_checkpointpool()

    return prepare_model, gen_input, run_model, teardown


def prepare_model(model_name, batch_size, use_dtr=False):
    if model_name.startswith('resnet') or model_name.startswith('tv_resnet'):
        return prepare_vision_cnn('resnet', model_name, batch_size, use_dtr)

    if model_name.startswith('densenet') or model_name.startswith('tv_densenet'):
        return prepare_vision_cnn('densenet' if model_name.startswith('densenet') else 'tv_densenet', model_name, batch_size, use_dtr)

    if model_name.startswith('inception'):
        return prepare_inception(model_name, batch_size, use_dtr)

    if model_name in {'transformer_encoder', 'lstm_encoder', 'gru_encoder'}:
        return prepare_word_language_model(model_name, batch_size, use_dtr)

    if model_name == 'treelstm':
        return prepare_treelstm(batch_size, use_dtr)

    if model_name == 'treelstm_old':
        return prepare_treelstm_old(batch_size, use_dtr)

    if model_name == 'transformer':
        return prepare_transformer(batch_size, use_dtr)

    if model_name == 'unet':
        return prepare_unet(batch_size, use_dtr)

    # our rewritten LSTM
    if model_name == 'lstm':
        return prepare_lstm_classification(batch_size, use_dtr)

    if model_name == 'unroll_gan':
        return prepare_unrolled_gan(batch_size, use_dtr)

    raise Exception('Model {} not supported'.format(model_name))
