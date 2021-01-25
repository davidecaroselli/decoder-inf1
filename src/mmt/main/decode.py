import argparse
import sys

from mmt import utils
from mmt.checkpoint import CheckpointRegistry
from mmt.decoder import ModelConfig, MMTDecoder


def main(argv=None):
    # Args parse
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Run a forever-loop serving translations')
    parser.add_argument('src_lang', help='the source language')
    parser.add_argument('tgt_lang', help='the target language')
    parser.add_argument('model', metavar='MODEL', help='the path to the decoder model')
    parser.add_argument('-g', '--gpu', dest='gpu', help='specify the GPU to use (default none)', default=None, type=int)

    args = parser.parse_args(argv)

    # Redirecting stdout and stderr to /dev/null
    # ------------------------------------------------------------------------------------------------------------------
    stdout, stderr = utils.mask_std_streams()

    # Setting up logging
    # ------------------------------------------------------------------------------------------------------------------
    utils.setup_basic_logging('INFO', stream=stderr)

    # Main loop
    # ------------------------------------------------------------------------------------------------------------------
    try:
        config = ModelConfig.load(args.model)

        builder = CheckpointRegistry.Builder()
        for name, checkpoint_path in config.checkpoints:
            builder.register(name, checkpoint_path)
        checkpoints = builder.build(args.gpu)

        decoder = MMTDecoder(checkpoints, device=args.gpu, tuning_ops=config.tuning)
    except Exception as e:
        stdout.write('ERROR: %s\n' % str(e))
        stdout.flush()
        raise

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break

            translations = decoder.translate(args.src_lang, args.tgt_lang, [line.rstrip()])
            stdout.write(translations[0].text + '\n')
            stdout.flush()
    except KeyboardInterrupt:
        pass  # ignore and exit
