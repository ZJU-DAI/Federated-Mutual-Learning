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
device = torch.device(args.device)
print("Running on", device)
data = Data(args)
logger = SummaryWriter('./logs')

# init nodes
global_node = Global_Node(args)
Node_List = []
for i in range(args.node_num):
    Node_List.append(Node(i, data.train_loader, args))

# init variables
recorder = Recorder(args)

# start
for rounds in range(args.R):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    LR_scheduler(rounds, Node_List, args)
    for i in range(len(Node_List)):
        Node_List[i].fork(global_node)
        for epoch in range(args.E):
            train(Node_List[i])
            recorder.validate(Node_List[i], data.test_loader)
        recorder.printer(Node_List[i])
    global_node.merge(Node_List)
recorder.finish()
