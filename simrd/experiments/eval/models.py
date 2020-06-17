import os

"""
Details of models for the simulated eval.
"""

LOG_PATH = 'models'

ALL_MODELS = {}
MODELS = {}  # ones with successful defaults

def register_model(fail=False):
  def F(f):
    if not fail:
      MODELS[f.__name__] = f
    ALL_MODELS[f.__name__] = f
    return f
  return F

def check_args(cfg, **kwargs):
  good = True
  for val in cfg.values():
    if val is None:
      good = False
      break
  good = good and os.path.isfile(cfg['log'])
  if not good:
    raise ValueError('Invalid model parameters for {}: {}'.format(
      cfg['name'], kwargs
    ))

@register_model()
def densenet(**kwargs):
  batch, layers = kwargs.get('batch','64'), kwargs.get('layers', '100')
  cfg = {
    'name': 'DenseNet-BC',
    'batch_size': str(batch),
    'layers': str(layers),
    'type': 'static',
    'log': LOG_PATH + '/densenet{}-{}.log'.format(batch, layers),
    'has_start': True
  }
  check_args(cfg, **kwargs)
  return cfg

@register_model()
def resnet(**kwargs):
  batch, layers = kwargs.get('batch', '32'), kwargs.get('layers', '32')
  cfg = {
    'name': 'ResNet',
    'batch_size': str(batch),
    'layers': str(layers),
    'type': 'static',
    'log': LOG_PATH + '/resnet{}-{}.log'.format(batch, layers),
    'has_start': True
  }
  check_args(cfg, **kwargs)
  return cfg

@register_model(fail=True)
def transformer(**kwargs):
  cfg = {
    'name': 'Transformer',
    'batch_size': None,
    'layers': None,
    'type': 'dynamic',
    'log': None,
    'has_start': True
  }
  if kwargs.get('fail', False):
    cfg['batch_size'] = '32'
    cfg['layers'] = 'default'
    cfg['log'] = LOG_PATH + '/failures/transformer-32-default.log'
  check_args(cfg, **kwargs)
  return cfg

@register_model(fail=True)
def gru(**kwargs):
  cfg = {
    'name': 'GRU',
    'batch_size': None,
    'layers': None,
    'type': 'dynamic',
    'log': None,
    'has_start': True
  }
  if kwargs.get('fail', False):
    cfg['batch_size'] = '32'
    cfg['layers'] = 'default'
    cfg['log'] = LOG_PATH + '/failures/gru_encoder-32-default.log'
  check_args(cfg, **kwargs)
  return cfg

@register_model()
def lstm(**kwargs):
  batch, layers = kwargs.get('batch', '32'), kwargs.get('layers', '1x')
  cfg = {
    'name': 'LSTM',
    'batch_size': str(batch),
    'layers': str(layers),
    'type': 'dynamic',
    'log': LOG_PATH + '/lstm-{}-{}.log'.format(batch, layers),
    'has_start': True
  }

  if kwargs.get('fail', False):
    cfg['batch_size'] = '32'
    cfg['layers'] = 'default'
    cfg['log'] = LOG_PATH + '/failures/lstm_encoder-32-default.log'

  check_args(cfg, **kwargs)
  return cfg

@register_model()
def treelstm(**kwargs):
  batch, layers = kwargs.get('batch', '32'), kwargs.get('layers', 'default')
  cfg = {
    'name': 'TreeLSTM',
    'batch_size': str(batch),
    'layers': str(layers),
    'type': 'dynamic',
    'log': LOG_PATH + '/treelstm-{}-{}.log'.format(batch, layers),
    'has_start': True,
  }
  if kwargs.get('fail', False):
    cfg['batch_size'] = '32'
    cfg['layers'] = 'default'
    cfg['log'] = LOG_PATH + '/failures/treelstm_old-32-default.log'
  check_args(cfg, **kwargs)
  return cfg

@register_model()
def unet(**kwargs):
  batch = kwargs.get('batch', '4')
  cfg = {
    'name': 'UNet',
    'batch_size': str(batch),
    'layers': 'default',
    'type': 'static',
    'log': LOG_PATH + '/unet-{}-default.log'.format(batch),
    'has_start': True
  }
  check_args(cfg, **kwargs)
  return cfg

@register_model()
def unrolled_gan(**kwargs):
  batch = kwargs.get('batch', 32)
  cfg = {
    'name': 'Unrolled-GAN',
    'batch_size': str(batch),
    'layers': 'default',
    'type': 'dynamic/meta',
    'log': LOG_PATH + '/unroll_gan-32.log',
    'has_start': True
  }
  check_args(cfg, **kwargs)
  return cfg
