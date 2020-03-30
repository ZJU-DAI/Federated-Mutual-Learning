import Train_mutual
import Train_normal
import Train_avg


def Trainer(args):
    if args.algorithm == 'fed_mutual':
        from Train_mutual import train
    elif args.algorithm == 'fed_avg':
        from Train_avg import train
    elif args.algorithm == 'normal':
        from Train_normal import train
