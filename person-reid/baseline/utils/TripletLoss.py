import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)


    # def forward(self, inputs, targets):
    #     """
    #     Args:
    #         inputs: feature matrix with shape (batch_size, feat_dim)
    #         targets: ground truth labels with shape (num_classes)
    #         return the loss
    #         # PLEASE WRITE THIS Method
    #     """
    #     n=inputs.size(0)
    #     # calculate the distance between pairs of inputs 
    #     # the distance=(a-b)^2=a^2+b^2-2ab
 
    #     dot_product=inputs.mm(inputs.t()) # ab
    #     square_norm=torch.diagonal(dot_product) #a^2 and b^2
    #     #a^2+b^2-2ab
    #     distance=square_norm.expand(1,n).t()-2.0*dot_product+square_norm.expand(1,n)
    #     # calculate the real distance
    #     distance=distance.clamp(min=1e-16).sqrt()

    #     # for i,j,k: i!=j!=k, 
    #     # create a mask, in which 
    #     # True means targets[i]==targets[j] and False means targets[i]!=targets[k]
    #     mask=targets.expand(n,n).eq(targets.expand(n,n).t())
    #     # find the hardest positive and negative
    #     positive_samples=[]
    #     negative_samples=[]
    #     for i in range(n):
    #         # hardest positive means positive but with maximum distance
    #         positive_samples.append(distance[i][mask[i]].max())
    #         # hardest negative means negative but with minimum distance
    #         negative_samples.append(distance[i][mask[i]==False].min())

    #     positive_samples = torch.Tensor(positive_samples).to("cuda")
    #     negative_samples = torch.Tensor(negative_samples).to("cuda")
        
    #     # the flag, if y=1 which means the first param is bigger than the second.
    #     y=torch.ones(positive_samples.size(),requires_grad=True).to("cuda")
    #     loss=self.ranking_loss(positive_samples,negative_samples,y)
    #     return loss

    def forward(self,inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_( inputs, inputs.t(),beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.Tensor(dist_ap).to("cuda")
        dist_an = torch.Tensor(dist_an).to("cuda")
        # Compute ranking hinge loss
        y=torch.ones(dist_ap.size(),requires_grad=True).to("cuda")
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss

if __name__=="__main__":
    triplet_loss=TripletLoss()
    pred=torch.Tensor([[1,1],[2,2],[1,1]])
    target=torch.Tensor([[2],[2],[1]])
    print(triplet_loss(pred,target))
    print(triplet_loss.loss_tri(pred,target))

