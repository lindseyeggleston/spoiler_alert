import h5py
import argparse as ap
from bidirectional_lm import BiLM


def parse_arguments():
    parser = ap.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--save_as", required=True)
    parser.add_argument("-n", "--neurons", type=check_positive,
                        help="number of neurons in every hidden layer")
    parser.add_argument("-b", "--batchsize", type=check_positive,  default=256,
                        help="size of batch for updating weights")
    parser.add_argument("-e", "--epochs", type=check_positive, default=1,
                        help="number of epochs for training")
    parser.add_argument("-a", "--alpha", type=check_percentage, default=.001,
                        help="learning rate for neural network")
    parser.add_argument("-d", "--dropout", type=check_percentage, default=.2,
                        help="dropout rate at each hidden layer")
    parser.add_argument("-m", "--max_seq_len", type=check_positive,
                        help="maximum length of sequence")
    parser.add_argument("-em", "--embed_size", type=check_positive,
                        default=300, help="size of word embedding vectors")
    parser.add_argument("-v", "--vocab_size", type=check_positive,
                        help="size of vocabulary (i.e max features)",
                        default=10000)
    args = parser.parse_args()
    return args


def check_positive(arg):
    value = int(arg)
    if value < 1:
        raise ap.ArgumentTypeError("Value should be greater than 0")
    return value


def check_percentage(arg):
    value = float(arg)
    if value < 0 or value > 1:
        raise ap.ArgumentTypeError("Value should be between 0 and 1")
    return value


# TODO: add h5 file reader
def read_data_from_file(filepath):
    pass


if __name__ == '__main__':
    args = parse_args()
    model = BiLM(n_neurons=args.neurons, max_seq_len=args.max_seq_len,
                 embed_size=args.embed_size, vocab_size=args.vocab_size,
                 dropout=args.dropout, alpha=args.alpha)
    # TODO: add training command
    # TODO: incorporate TensorBoard visualizations
    # TODO: add save_as
