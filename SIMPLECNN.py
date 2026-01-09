import torch 
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import transforms



# shape -> (input + 2 padding - kernel) / stride + 1
class SimpleCNN(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(inchannel, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier will be created on first forward pass
        self.classifier = None
        self.flattened_size = None
    
    def forward(self, x):
        x = self.features(x)
        
        # Create classifier on first pass if not created
        if self.classifier is None:
            self.flattened_size = x.numel() // x.shape[0]
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flattened_size, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 10)
            ).to(x.device)
            
        
        x = self.classifier(x)
        return x
    
model = SimpleCNN(inchannel = 3)


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
