import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
import torch.nn.functional as F

num_classes = 751  # change this depend on your dataset
num_embedding=1024

class REID_NET(nn.Module):
    def __init__(self):
        # write the CNN initialization
        super(REID_NET, self).__init__()

        self.fc_id= nn.Linear(num_embedding, num_classes)
        self.fc_metric=nn.Linear(512 * 4, num_embedding)
        self.resnet = resnet50(pretrained=True)

    def forward(self, x):
        # write the CNN forward
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        predict_id= self.fc_id(x) 
        predict_metric= self.fc_metric(x)         

        predict = torch.cat([predict_id, predict_metric], dim=1)

        return predict,  predict_metric,predict_id,

if __name__=="__main__":
    img=torch.randn((2,3,384, 128))
    net=REID_NET()
    pred=net(img)
    print(pred)
    