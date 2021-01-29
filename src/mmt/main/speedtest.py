import argparse
import time

from mmt import utils
from mmt.checkpoint import CheckpointRegistry
from mmt.decoder import Suggestion, ModelConfig, MMTDecoder

TEST_TEXT = 'Companies and LSPs can translate their content with the ModernMT service in many languages ' \
            'directly on their production environment thanks to our simple RESTful API .'


def translate_test(decoder, adaptive=True):
    suggestions = [
        Suggestion('en', 'it', 'We offer a simple RESTful API', 'Offriamo una semplice API di tipo REST', 1.0),
        Suggestion('en', 'it', 'The production environment is running', 'Il sistema di produzione è in esecuzione', 1.0)
    ]

    # Force full reset
    decoder._nn_needs_reset = True

    return decoder.translate('en', 'it', [TEST_TEXT], suggestions=suggestions if adaptive else None)[0]


def main(argv=None):
    # Args parse
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Run a forever-loop testing translation speed')
    parser.add_argument('model', metavar='MODEL', help='the path to the decoder model')
    parser.add_argument('-g', '--gpu', dest='gpu', metavar='GPU', type=int, default=0,
                        help='the id of the GPU to use (default: 0)')
    parser.add_argument('-a', '--adaptive', dest='adaptive', action='store_true', default=False,
                        help='test with adaptive feature')

    args = parser.parse_args(argv)

    # Setting up logging
    # ------------------------------------------------------------------------------------------------------------------
    logger = utils.setup_basic_logging()

    # Main loop
    # ------------------------------------------------------------------------------------------------------------------
    try:
        config = ModelConfig.load(args.model)

        begin_ts = time.time()
        builder = CheckpointRegistry.Builder()
        for name, checkpoint_path in config.checkpoints:
            builder.register(name, checkpoint_path)
        checkpoints = builder.build(args.gpu)
        logger.info('[1/2] Loaded %d checkpoints in %.1fs' % (len(checkpoints), time.time() - begin_ts))

        begin_ts = time.time()
        device=args.gpu if args.gpu >= 0 else None
        decoder = MMTDecoder(checkpoints, device=device, tuning_ops=config.tuning)
        logger.info('[2/2] Decoder created in %.1fs' % (time.time() - begin_ts))

        while True:
            translation = translate_test(decoder, adaptive=args.adaptive)

            print(translation.text)
            print(translation.alignment)
    except KeyboardInterrupt:
        pass  # ignore and exit


if __name__ == '__main__':
    main()
