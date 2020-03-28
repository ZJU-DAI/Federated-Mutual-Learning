Node_List = [1, 2, 3, 8, 9]


def Node_do(List, index):
    if index != len(List):
        print(List[index])
        Node_do(List, index + 1)
    else:
        return


Node_do(Node_List, 0)
