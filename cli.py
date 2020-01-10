#https://www.sicara.ai/blog/2018-12-18-perfect-command-line-interfaces-python
#https://pypi.org/project/python-dotenv/
import click

@click.command()
@click.argument('config_path', nargs=-1)
@click.option('--training/--inference', '-t/-i')
def execute(config_path, training, key):
    #1. load config from path
    pass

def training():
    pass

def inference():
    pass

if __name__ == '__main__':
    execute()