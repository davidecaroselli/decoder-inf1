import torch
from fairseq.models.transformer import TransformerModel

TEST_TEXT = 'Companies and LSPs can translate their content with the ModernMT service in many languages ' \
            'directly on their production environment thanks to our simple RESTful API .'

class MyGen(torch.nn.Module):
    def __init__(self, model):
        super(MyGen, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.translate(x)

en2de = TransformerModel.from_pretrained(
  'checkpoints/fconv_wmt17_en_de',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/wmt17_en_de',
  bpe='subword_nmt',
  bpe_codes='examples/translation/wmt17_en_de/code'
)

print(type(en2de))
# sample = TEST_TEXT
# print(f'- Using [MyGen.forward]: {en2de.translate(sample)}')

# my_gen = MyGen(en2de)
# print(my_gen(TEST_TEXT))
# print(f'- Using [MyGen.forward]: {my_gen.forward(sample)}')
# torch.jit.trace(my_gen, 'Hello')