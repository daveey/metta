import sys

from sample_factory.enjoy import enjoy

import train as train

def main():
    """Script entry point."""
    train.register_custom_components()
    cfg = train.parse_custom_args(evaluation=True)
    status = enjoy(cfg)
    return status[0]


if __name__ == "__main__":
    sys.exit(main())
