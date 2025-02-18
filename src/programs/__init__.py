import click
from pathlib import Path

@click.group()
def cli():
    """Gaussian Tools - Modern Python implementation of molecular modeling tools"""
    pass

@cli.command()
@click.argument('input_file')
@click.argument('mode', type=int)
@click.argument('amplitude', type=float)
@click.argument('output_file')
def displace(input_file: str, mode: int, amplitude: float, output_file: str):
    """Generate displaced structure along normal mode"""
    from .displace import Displace
    program = Displace(input_file)
    program.run(mode, amplitude, output_file)

@cli.command()
@click.argument('root_name')
@click.argument('input_files', nargs=-1)
def dushin(root_name: str, input_files: tuple):
    """Analyze force constants and calculate frequencies"""
    from .dushin import Dushin
    program = Dushin(root_name)
    for i, file in enumerate(input_files):
        program.process_gaussian_output(file, is_reference=(i==0))

@cli.command()
@click.argument('ref_file')
@click.argument('comp_file')
@click.argument('output_file')
def compare(ref_file: str, comp_file: str, output_file: str):
    """Compare molecular geometries"""
    from .compare_geom import CompareGeom
    program = CompareGeom()
    program.read_geometries(ref_file, comp_file)
    program.write_comparison_report(output_file)

if __name__ == '__main__':
    cli()