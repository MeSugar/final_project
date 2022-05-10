import click

from . import __version__

@click.command()
@click.version_option(version=__version__)
def main():
    """Forest competition"""
    click.print("Hello, world!")

def hi():
    click.echo("Hi, world!")