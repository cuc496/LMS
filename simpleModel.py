import torch.nn as nn
import torch
"""
class FashionSimpleNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x
"""
class Swish(nn.Module):
  def forward(self, input):
    return (input * torch.sigmoid(input))
  
  def __repr__(self):
    return self.__class__.__name__ + ' ()'

class FashionSimpleNet(nn.Module):
    def __init__ (self):
        super(FashionSimpleNet,self).__init__()
        
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.swish1=Swish()
        nn.init.xavier_normal_(self.cnn1.weight)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=1)
        
        self.cnn2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(32)
        self.swish2=Swish()
        nn.init.xavier_normal_(self.cnn2.weight)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)

        self.cnn3=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(64)
        self.swish3=Swish()
        nn.init.xavier_normal_(self.cnn3.weight)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64*6*6,10)
        
        self.softmax=nn.Softmax(dim=1)
        
        
    def forward(self,x):
        out=self.cnn1(x)
        out=self.bn1(out)
        out=self.swish1(out)
        out=self.maxpool1(out)
        out=self.cnn2(out)
        out=self.bn2(out)
        out=self.swish2(out)
        out=self.maxpool2(out)
        out=self.cnn3(out)
        out=self.bn3(out)
        out=self.swish3(out)
        out=self.maxpool3(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        out=self.softmax(out)
        
        return out
    
    
class CNN2(nn.Module):
    def __init__ (self):
        super(CNN2,self).__init__()
        
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.swish1=Swish()
        nn.init.xavier_normal_(self.cnn1.weight)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=1)
        
        self.cnn2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(32)
        self.swish2=Swish()
        nn.init.xavier_normal_(self.cnn2.weight)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        """
        self.cnn3=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(64)
        self.swish3=Swish()
        nn.init.xavier_normal_(self.cnn3.weight)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)
        """
        self.fc1 = nn.Linear(32*13*13,10)
        
        self.softmax=nn.Softmax(dim=1)
        
        
    def forward(self,x):
        out=self.cnn1(x)
        out=self.bn1(out)
        out=self.swish1(out)
        out=self.maxpool1(out)
        out=self.cnn2(out)
        out=self.bn2(out)
        out=self.swish2(out)
        out=self.maxpool2(out)
        #out=self.cnn3(out)
        #out=self.bn3(out)
        #out=self.swish3(out)
        #out=self.maxpool3(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        out=self.softmax(out)
        
        return out
#https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844
#nn.Conv2d(input channel, output channel, filter size)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
            
class simplerCNN2(nn.Module):
    def __init__ (self):
        super(simplerCNN2,self).__init__()
        
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=8, kernel_size=3,stride=1,padding=1)
        #self.bn1=nn.BatchNorm2d(16)
        self.ReLU1=nn.ReLU(inplace=True)
        #nn.init.xavier_normal_(self.cnn1.weight)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=1)
        
        self.cnn2=nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=1,padding=1)
        #self.bn2=nn.BatchNorm2d(32)
        self.ReLU2=nn.ReLU(inplace=True)
        #nn.init.xavier_normal_(self.cnn2.weight)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        """
        self.cnn3=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(64)
        self.swish3=Swish()
        nn.init.xavier_normal_(self.cnn3.weight)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)
        """
        self.fc1 = nn.Linear(16*13*13,10)
        
        self.softmax=nn.Softmax(dim=1)
        
        
    def forward(self,x):
        out=self.cnn1(x)
        out=self.ReLU1(out)
        out=self.maxpool1(out)
        out=self.cnn2(out)
        out=self.ReLU2(out)
        out=self.maxpool2(out)
        #out=self.cnn3(out)
        #out=self.bn3(out)
        #out=self.swish3(out)
        #out=self.maxpool3(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        out=self.softmax(out)
        
        return out