import copy
import torch
import Model


def init_model(model_type):
    model = []
    if model_type == 'LeNet5':
        model = Model.LeNet5()
    elif model_type == 'CNNCifar':
        model = Model.CNNCifar()
    elif model_type == 'ResNet18':
        model = Model.ResNet18()
    return model


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer


# def init_scheduler(optimizer, args):
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
#     return scheduler


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Node(object):
    def __init__(self, num, local_data, args):
        self.args = args
        self.num = num + 1
        self.device = self.args.device
        self.local_data = local_data
        self.model = init_model(self.args.local_model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)
        # self.scheduler = init_scheduler(self.optimizer, self.args)
        self.meme = init_model(self.args.global_model).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args)
        # self.meme_scheduler = init_scheduler(self.meme_optimizer, self.args)

    def fork(self, global_node):
        self.meme = copy.deepcopy(global_node.model).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args)
        # self.meme_scheduler = init_scheduler(self.meme_optimizer, self.args)


class Global_Node(object):
    def __init__(self, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        # self.meme = init_model(self.args.global_model).to(self.device)
        self.model = init_model(self.args.global_model).to(self.device)
        self.Dict = self.model.state_dict()

    def merge(self, Node_List):
        weights_zero(self.model)
        Node_State_List = []
        for i in range(len(Node_List)):
            Node_State_List.append(copy.deepcopy(Node_List[i].meme.state_dict()))
        for key in self.Dict.keys():
            for i in range(len(Node_List)):
                self.Dict[key] += Node_State_List[i][key]
            self.Dict[key] /= len(Node_List)
            # self.Dict[key] = self.Dict[key] / len(Node_List) #IT'S NOT WORK!!!

    # def merge(self, Node_List):
    #     Node_State_List = []
    #     for i in range(len(Node_List)):
    #         Node_State_List.append(copy.deepcopy(Node_List[i].meme.state_dict()))
    #     weights = copy.deepcopy(Node_State_List[0])
    #     for key in weights.keys():
    #         for i in range(1, len(Node_State_List)):
    #             weights[key] += Node_State_List[i][key]
    #         weights[key] = torch.div(weights[key], len(Node_List))
    #     self.model.load_state_dict(weights)

    # def merge(self, Node_List):
    #     weights_zero(self.model)
    #     Node_State_List = []
    #     for i in range(len(Node_List)):
    #         Node_State_List.append(copy.deepcopy(Node_List[i].meme.state_dict()))
    #     global_state_dict = copy.deepcopy(self.Dict)
    #     for key in global_state_dict.keys():
    #         for i in range(len(Node_List)):
    #             global_state_dict[key] += Node_State_List[i][key]
    #         global_state_dict[key] = torch.div(global_state_dict[key], len(Node_List))
    #     self.model.load_state_dict(global_state_dict)
