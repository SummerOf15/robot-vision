from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
import torch

class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)
        labels=labels.long()

        Triplet_Loss = triplet_loss(outputs[1], labels)
        #Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        # CrossEntropy_Loss = cross_entropy_loss(outputs[2], labels)
        #CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        # loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        # print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.4f' % (
        #     loss_sum.data.cpu().numpy(),
        #     Triplet_Loss.data.cpu().numpy(),
        #     CrossEntropy_Loss.data.cpu().numpy()),
        #       end=' ')
        return TripletLoss

if __name__=="__main__":
    cross_entropy_loss = CrossEntropyLoss()
    a=torch.rand((16,140))
    b=torch.rand((16)).long()
    print(cross_entropy_loss(a,b))