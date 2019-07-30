import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)



"""
IMPLEMENT THE MODEL YOU SEE IN THE
"Siamese Neural Networks for One-shot Image Recognition" paper
Dubbed SalakhNet as in Salakhutdinov, the main author of the paper.
"""

class SalakhNet(nn.Module):
    def __init__(self):
        super(SalakhNet, self).__init__()
        """
        TASK FOR MICHAEL:Your layer definitions come here
        """
        self.convnet = nn.Sequential(
            nn.AdaptiveAvgPool2d((105, 105)),
            nn.Conv2d(3, 64, 10, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7, stride = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4, stride = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4, stride = 1), 
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            # Create 4096-element feature vector
            nn.Linear(256 * 6 * 6, 4096), nn.Sigmoid()
        )

    def forward(self, x):
        """
        TASK FOR MICHAEL:Your forward computations come here
        """
        #print(x)
        # Create feature maps using CNN
        output = self.convnet(x)
        # Flatten feature maps
        output = output.view(output.size()[0], -1)
        #print(output)
        # Feed flattened vector into fully connected layer
        output = self.fc(output)
        
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net()
        self.alpha = torch.rand(4096).to('cuda:0')

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        
        #print("Feature Vector 1:", output1)
        #print("Feature Vector 2:", output2)
        
        distance = torch.abs(output1 -  output2).squeeze(0)
        #print(distance.shape)
        #distance = self.fc_fuse(distance)
        distance = torch.sigmoid(torch.dot(self.alpha, distance)) 
        return distance

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

# Test whether SalakhNet works
if __name__ == "__main__":
    """
    net = SalakhNet()
    tensor = torch.rand(105, 105).unsqueeze(0).unsqueeze(0)
    print(tensor.size())
    output = net(tensor)
    print("Feature Vector of size:", output.size())
    print(output)
    """
    
    net = SiameseNet(SalakhNet)
    """
    for i, (image_1, image_2, label_1, label_2) in enumerate(trainloader):
        tensor1 = image_1
        tensor2 = image_2
        distance = net(tensor1, tensor2)
        print("Scalarized distance:", distance, "\n\n")
        if i > 10:
            break"""
    
    for i in range(10):
        print("Test", i + 1)
        tensor1 = torch.rand(105, 105).unsqueeze(0).unsqueeze(0)
        tensor2 = torch.rand(105, 105).unsqueeze(0).unsqueeze(0)
        
        print("Shape:", tensor1.shape)
        
        zerotensor = torch.zeros(105,105).unsqueeze(0).unsqueeze(0)
        onestensor = torch.ones(105,105).unsqueeze(0).unsqueeze(0)
        #print(zerotensor.shape)
        #print(onestensor.shape)
        #print("Tensor 1:", tensor1)
        #print("Tensor 2:", tensor2)

        distance = net(zerotensor, onestensor)
        print("Scalarized distance:", distance, "\n\n")