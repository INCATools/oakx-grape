"""Command line interface for oakx-grape."""
import click
import logging

from oakx_grape import __version__
from oakx_grape.grape_implementation import GrapeImplementation

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)

@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """CLI for oakx-grape.

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    if quiet:
        logger.setLevel(level=logging.ERROR)

@main.command()
def run():
    """Run the oakx-grape's demo command."""
    impl = GrapeImplementation()
     


if __name__ == "__main__":
    main()
