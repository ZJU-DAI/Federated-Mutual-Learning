import copy
import torch
import Model


class Node(object):
    def __init__(self, num, model, local_data, device):
        self.num = num
        self.device = device
        self.local_data = local_data
        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.meme = Model.ResNet18().to(self.device)
        self.meme_optimizer = torch.optim.SGD(self.meme.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        self.meme_scheduler = torch.optim.lr_scheduler.StepLR(self.meme_optimizer, step_size=30, gamma=0.1)

    def fork(self, global_node):
        self.meme = copy.deepcopy(global_node.model).to(self.device)
        self.meme_optimizer = torch.optim.SGD(self.meme.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
        self.meme_scheduler = torch.optim.lr_scheduler.StepLR(self.meme_optimizer, step_size=30, gamma=0.1)


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Global_Node(object):
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)
        self.Dict = self.model.state_dict()

    def merge(self, Node_List):
        weights_zero(self.model)
        Node_State_List = []
        for i in range(len(Node_List)):
            Node_State_List.append(copy.deepcopy(Node_List[i].meme.state_dict()))
        for key in self.Dict.keys():
            for i in range(len(Node_List)):
                self.Dict[key] += Node_State_List[i][key]
            self.Dict[key] /= 2
            # self.Dict[key] = self.Dict[key] / 2 #IT'S NOT WORK!!!

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
