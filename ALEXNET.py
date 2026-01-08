import torch 
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import transforms

class AlexNET(nn.Module):
    def __init__(self , inchannel , height , width):
        super().__init__()
        self.feature = nn.Sequential(
            # ( inchannel , height , width) -> ( 96 , height // 4 , width // 4)
            nn.Conv2d(inchannel , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
            nn.ReLU(inplace=True),

            # ( 96 , height // 4 , width // 4) -> ( 64 , height // 8 , width // 8)
            nn.MaxPool2d(2 , stride = 2) ,

            nn.BatchNorm2d(64) ,

            # ( 96 , height // 8 , width // 8) -> ( 128 , height // 8 , width // 8)
            nn.Conv2d(64 , 128 , kernel_size = 3 , padding = 2) ,
            nn.ReLU(inplace=True),

            # ( 128 , height // 8 , width // 8) -> ( 128 , height // 16 , width // 16)
            nn.MaxPool2d(2 , stride = 2) ,

            nn.BatchNorm2d(128) ,

            # ( 128 , height // 16 , width // 16) -> ( 192 , height // 16 , width // 16)
            nn.Conv2d(128 , 192 , kernel_size = 2 , stride = 2 ) ,
            nn.ReLU(inplace=True),

            # ( 192 , height // 16 , width // 16) -> ( 192 , height // 16 , width // 16)
            nn.Conv2d(192 , 192 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(inplace=True),

            # ( 192 , height // 16 , width // 16) -> ( 256 , height // 32 , width // 32)
            nn.Conv2d(192 , 256 , kernel_size = 3 , padding = 1) ,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        with torch.no_grad():
            dummy = torch.randn(1 , 3 , 32 , 32)
            output = self.feature(dummy)

            self.flat_size = output.numel() // output.shape[0]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  
            nn.Linear(self.flat_size, 1024),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),  
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
           
            nn.Linear(1024, 512)
        )

    def forward(self , x):
        x = self.feature(x)
        x = x.view(x.size(0) , -1)  #flat
        x = self.classifier(x)

        return x
    

model = AlexNET(inchannel = 3 , height = 227 , width = 227)


# we need to define how the data should be like in normalized form and in tensor form
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean
                         (0.2470, 0.2435, 0.2616))  # std
])

# load data
traindata = torchvision.datasets.CIFAR10(root = '/data' , download = True , train = True , transform = transform)
testdata = torchvision.datasets.CIFAR10(root = '/data' , download = True , train = False , transform = transform)

# load the data
trainloader = torch.utils.data.DataLoader(traindata , batch_size = 64 , shuffle = True)
testloader = torch.utils.data.DataLoader(testdata , batch_size = 64 , shuffle = False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

EPOCH = 10
for epoch in range(EPOCH):
    model.train()
    runtimeLOSS = 0
    correct = 0
    totalLOSS = 0

    for i , data in enumerate(trainloader):
        images , labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output , labels)

        loss.backward()
        optimizer.step()

        runtimeLOSS += loss.item()
        _ , pred = output.max(1)
        totalLOSS += labels.size(0)
        correct += pred.eq(labels).sum().item()
    trainacc = correct * 100 / totalLOSS
    print(f"Epoch [{epoch+1}/{EPOCH}] Loss: {runtimeLOSS/len(trainloader):.4f} | Acc: {trainacc:.2f}%")

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for i , dataa in enumerate(testloader):
            input , label = dataa
            input , label = input.to(device), label.to(device)
            output = model(input)

            _ , predi = output.max(1)
            test_total += label.size(0)
            test_correct += predi.eq(label).sum().item()
    
    print(f"Test Accuracy: {100. * test_correct / test_total:.2f}%")
    print("-" * 30)