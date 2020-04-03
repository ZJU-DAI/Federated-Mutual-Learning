import torch
from Node import Node, Global_Node
from torch.utils.tensorboard import SummaryWriter
from options import args_parser
from Data import Data
from utils import LR_scheduler, Recorder

# init args
args = args_parser()
if args.algorithm == 'fed_mutual':
    from Train_mutual import train
elif args.algorithm == 'fed_avg':
    from Train_avg import train
elif args.algorithm == 'normal':
    from Train_normal import train
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Running on', args.device)
data = Data(args)
logger = SummaryWriter('logs')

# init nodes
global_node = Global_Node(args)
Node_List = []
for k in range(args.node_num):
    Node_List.append(Node(k, data.train_loader[k], args))

# init variables
recorder = Recorder(args)
writer = SummaryWriter('logs')

# start
for rounds in range(args.R):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    LR_scheduler(rounds, Node_List, args)
    for k in range(len(Node_List)):
        Node_List[k].fork(global_node)
        for epoch in range(args.E):
            train(Node_List[k])
            recorder.validate(Node_List[k], data.test_loader)
            writer.add_scalar('Loss/validate', recorder.val_acc[str(Node_List[k].num)][-1], recorder.counter)
            # writer.add_scalars('Loss/validate', {'train': acc, 'val': recorder.acc[k]}, recorder.counter)
        recorder.printer(Node_List[k])
    global_node.merge(Node_List)
    recorder.validate(global_node, data.test_loader)
    recorder.printer(global_node)
recorder.finish()
