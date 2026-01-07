import torch 
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import transforms
import torch.optim as optim


# model 
class InceptionModule(nn.Module):
    def __init__(self , inchannel , a , b1 , b2 , c1 , c2 , d):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannel , a , kernel_size = 1 , padding = 0) 

        self.conv21 = nn.Conv2d(inchannel , b1 , kernel_size = 1 , padding = 0)
        self.conv2 = nn.Conv2d(b1 , b2 , kernel_size = 3 , padding = 1) 

        self.conv31 = nn.Conv2d(inchannel , c1 , kernel_size = 1 , padding = 0)
        self.conv3 = nn.Conv2d(c1 , c2 , kernel_size = 5 , padding = 2) 

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.convpool = nn.Conv2d(inchannel, d, kernel_size=1)

    def forward(self , x):

       out1 = F.relu(self.conv1(x))

       out2 = F.relu(self.conv21(x))
       out2 = F.relu(self.conv2(out2))

       out3 = F.relu(self.conv31(x))
       out3 = F.relu(self.conv3(out3))

       out4 = self.pool1(x)
       out4 = F.relu(self.convpool(out4))

       return torch.cat([out1 , out2 , out3 , out4] , dim = 1)
    
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024), 
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class GoogleNET(nn.Module):
    def __init__(self , num_classes = 1000 , training = True):
        super().__init__()
        self.training = training

        self.stem = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(True),
            nn.MaxPool2d(3 , stride = 2 , padding = 1) ,
            nn.ReLU(True),
            nn.Conv2d(64 , 192 , kernel_size = 3 , padding = 1),
            nn.ReLU(True),
            nn.MaxPool2d(3 , stride = 2 , padding = 1)
        )

        self.inception1 = InceptionModule(192 , 64 , 96 , 128 , 16 , 32 , 32)
        self.inception2 = InceptionModule(256 , 128, 128, 192 , 32 , 96 , 64)
        self.inception3 = InceptionModule(480 , 192, 96 , 208 , 16 , 48 , 64)
        self.inception4 = InceptionModule(512 , 160 , 112 , 224 , 24 , 64 , 64)
        self.inception5 = InceptionModule(512 , 128 , 128 , 256 , 24 , 64 , 64)
        self.inception6 = InceptionModule(512 , 112 , 144 , 288 , 32 , 64 , 64)
        self.inception7 = InceptionModule(528 , 256 , 160 , 320 , 32 , 128 , 128)
        self.inception8 = InceptionModule(832 , 256 , 160 , 320 , 32 , 128 , 128)
        self.inception9 = InceptionModule(832 , 384 , 192 , 384 , 48 , 128 , 128)

        self.aux1 = InceptionAux(512, num_classes)  
        self.aux2 = InceptionAux(528, num_classes)  

        self.avgpool = nn.AdaptiveAvgPool2d((1 , 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024 , num_classes)

    def forward(self , x):
        x = self.stem(x)

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        if self.training:
            aux1_out = self.aux1(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        if self.training:
            aux2_out = self.aux2(x)
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.inception9(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        main_out = self.fc(x)

        if self.training:
            return main_out, aux1_out, aux2_out
        return main_out

if __name__ == "__main__":
# how we need data (form normalized and in tensor form)
    transformss = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5) )
    ])

    # trainset and testset
    trainset = torchvision.datasets.CIFAR10(root = "./data" , train = True , download = True , transform = transformss)
    testset = torchvision.datasets.CIFAR10(root = "./data" , train = False , download = True , transform = transformss)

    # loading dataset (both train and test)
    trainloader = torch.utils.data.DataLoader(trainset , batch_size = 64 , shuffle = True , num_workers = 2 ) 
    testloader = torch.utils.data.DataLoader(testset , batch_size = 64 , shuffle = False , num_workers = 2 ) 
    EPOCH = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GoogleNET(num_classes = 10).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(EPOCH):
        model.train()
        runningloss = 0.0
        correct = 0
        total = 0

        for i , (images , labels) in enumerate(trainloader):
            images , labels = images.to(device = device) , labels.to(device = device)
            optimizer.zero_grad()

            mainout , aux1 , aux2 = model(images)

            loss0 = criterion(mainout , labels)
            loss1 = criterion(aux1 , labels)
            loss2 = criterion(aux2 , labels)

            loss = loss0 + 0.3*loss1 + 0.3*loss2

            loss.backward()
            optimizer.step()

            runningloss += loss.item()

            _, predicted = mainout.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCH}] Loss: {runningloss/len(trainloader):.4f} | Acc: {train_acc:.2f}%")

        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for input , labels in testloader:
                input , labels = input.to(device = device) , labels.to(device = device)

                output = model(input)

                _ , predicted = output.max(1)

                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        print(f"Test Accuracy: {100. * test_correct / test_total:.2f}%")
        print("-" * 30)


    # don't use  cifar 10 its 32 * 32 , and google net is for high res images like 227 * 227