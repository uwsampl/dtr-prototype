import os
import json

"""
Details of models for the simulated eval.
"""

LOG_PATH = 'logs'
MANIFEST_PATH = LOG_PATH + '/manifest.json'

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

MODELS = {}

print('loading manifest {}...'.format(MANIFEST_PATH))
MANIFEST = json.load(open(MANIFEST_PATH, 'r'))
MANIFEST = {model['name']: model for model in MANIFEST['models']}

INVALID_MODELS = []
for model_name, model in MANIFEST.items():
  try:
    check_args(model)
  except ValueError:
    print('ignoring invalid model "{}" from manifest'.format(model_name, MANIFEST_PATH))
    INVALID_MODELS.append(model_name)

for m in INVALID_MODELS:
  MANIFEST.pop(m)

print('found models: {}'.format(list(MANIFEST.keys())))
