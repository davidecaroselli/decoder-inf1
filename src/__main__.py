import argparse
import sys


def main():
    from mmt.main import decode, speedtest

    actions = {
        'decode': decode.main,
        'speedtest': speedtest.main,
    }

    parser = argparse.ArgumentParser(usage='%(prog)s [-h] ACTION [args]')
    parser.add_argument('action', metavar='ACTION', choices=actions.keys(), help='{%(choices)s}', nargs='?')

    argv = sys.argv[1:]

    if len(argv) == 0:
        parser.print_help()
        exit(1)

    command = argv[0]
    args = argv[1:]

    if command in actions:
        actions[command](args)
    else:
        parser.print_help()
        exit(1)


if __name__ == '__main__':
    main()
