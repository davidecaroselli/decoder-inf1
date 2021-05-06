import importlib
import sys
sys.path.append('src')

import torch
import torch.neuron

from mmt import utils
from mmt.checkpoint import CheckpointRegistry
from mmt.decoder import Suggestion, ModelConfig, MMTDecoder

TEST_TEXT = 'Companies and LSPs can translate their content with the ModernMT service in many languages ' \
            'directly on their production environment thanks to our simple RESTful API .'
MODEL_DIR = 'model'
device=None

config = ModelConfig.load(MODEL_DIR)
builder = CheckpointRegistry.Builder()
for name, checkpoint_path in config.checkpoints:
    builder.register(name, checkpoint_path)
checkpoints = builder.build(device)
decoder = MMTDecoder(checkpoints, device=device)

# A simple translation without using a tuner () 
trans_1 = decoder.translate('en', 'it', [TEST_TEXT])[0]
print(f'- Using [decoder.translate]: {trans_1.text}')
trans_2 = decoder._decode('en', 'it', [TEST_TEXT])[0]
print(f'- Using [decoder._decode]: {trans_2.text}')
trans_3 = decoder._decode_without_explicit_model('en', 'it', [TEST_TEXT])[0]
is_equal = "==" if trans_1.text == trans_3.text else "!="
print(f'- Using [decoder._translator._generate]: {trans_3.text}')
print('------')
print('Output of [decoder.translate]:')
print(f' {"==" if trans_1.text == trans_2.text else "!="} [decoder._decode]')
print(f' {"==" if trans_1.text == trans_3.text else "!="} [decoder._translator._generate]')

sample, input_indexes, sentence_len = decoder._make_decode_batch([TEST_TEXT], prefix_lang=None)

class MyGen(torch.nn.Module):
    def __init__(self, decoder):
        super(MyGen, self).__init__()
        self.decoder = decoder

    def forward(self, x):
        return self.decoder._decode_from_sample(x)
        
# Attempt to trace
print('Compiling...')
my_gen = MyGen(decoder)
print(f'- Using [MyGen.forward]: {my_gen.forward(sample)}')

compiled = torch.jit.trace(my_gen, sample)
compiled.save("compiled/MyGen_torch.pt")

# compiled = torch.neuron.trace(my_gen, sample)
# compiled.save("compiled/MyGen_neuron.pt")

print(f'- Using [MyGen.forward]: {my_gen.forward(sample)}')
print(f'- Using [pytorch.jit.trace]: {compiled(sample)}')
      

