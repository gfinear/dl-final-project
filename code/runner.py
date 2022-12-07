import argparse

from img2plot import Img2Plot


def main(args):
    img2plot = Img2Plot()
    plot = img2plot(args.image_filepath)

    if args.print is None:
        print(plot)
    else:
        with open(args.print, 'w') as output:
            output.write(plot)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_filepath', help='Filepath of image to summarize')
    parser.add_argument('-p', '--print', help='File to print to')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())