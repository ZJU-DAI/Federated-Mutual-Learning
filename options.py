import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='fed_mutual',
                        help="Type of algorithms:{fed_mutual, fed_avg, normal}")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="device: {cuda, cpu}")
    parser.add_argument('--node_num', type=int, default=3,
                        help="Number of nodes")
    parser.add_argument('--R', type=int, default=50,
                        help="Number of rounds: R")
    parser.add_argument('--E', type=int, default=5,
                        help="Number of local epochs: E")

    # Model
    parser.add_argument('--global_model', type=str, default='LeNet5',
                        help='Type of global model: {LeNet5, CNNCifar, ResNet18}')
    parser.add_argument('--local_model', type=str, default='LeNet5',
                        help='Type of local model: {LeNet5, CNNCifar, ResNet18}')

    # Data
    parser.add_argument('--batchsize', type=int, default=128,
                        help="batchsize")

    # Optim
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='learning rate decay step size')
    parser.add_argument('--stop_decay', type=int, default=30,
                        help='round when learning rate stop decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')

    args = parser.parse_args()
    return args
