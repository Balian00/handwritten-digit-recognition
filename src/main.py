# import
import click
from train import train
from test import test, test_one

# click
@click.command()
@click.option('--mode', type=click.Choice(['train', 'test', 'test_one']))
@click.option('--image', type=str, default=None)

# main
def main(mode, image=None) :
    print(f'Code running in {mode} mode...')
    if mode == 'train' :
        train()
    elif mode == 'test' :
        test()
    elif mode == 'test_one' :
        test_one(image)
    else :
        return 0
    
if __name__ == '__main__':
    main()