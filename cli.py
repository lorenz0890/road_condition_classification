#https://www.sicara.ai/blog/2018-12-18-perfect-command-line-interfaces-python
#https://pypi.org/project/python-dotenv/
import sys
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import click
import json
from pipeline.pipeline_facade import ConcretePipelineFacade

@click.command()
@click.argument('config_path', nargs=-1)
@click.option('--training/--inference', '-t/-i')
def execute_command(config_path, training):
    # 1. Load config from path
    config = load_config(config_path[0])
    print('--------------------LOAD CONFIG--------------------')
    print(json.dumps(config, indent=1))

    # 2. Execute training or Inference
    pipeline = ConcretePipelineFacade()
    if training:
        pipeline.execute_training(config)
    else:
        pipeline.execute_inference(config)

def load_config(config_path):
    config = None
    with open(config_path) as json_file:
        config = json.load(json_file)

    return config #throw error of data is None

if __name__ == '__main__':
    execute_command()