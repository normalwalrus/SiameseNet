from facenet_pytorch import InceptionResnetV1
import torch
import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self, model_choice):
        super().__init__()

        if model_choice == 1:
            self.encoder = InceptionResnetV1(pretrained='vggface2').eval()
        elif model_choice == 2:
            self.encoder = InceptionResnetV1(pretrained='casia-webface').eval()

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        emb_len = 512
        self.last = nn.Sequential(
            nn.Linear(2*emb_len, 200, bias=False), 
            nn.BatchNorm1d(200, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, input1, input2):
        
        emb1 = self.encoder(input1)
        emb2 = self.encoder(input2)
        # Feature extraction2
        
        x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)
        x2 = torch.pow(emb1 - emb2, 2)
        #x3 = emb1 * emb2
        #x4 = emb1 + emb2
        #x5 = emb FE2
        
        x = torch.cat((x1,x2), dim=1) # Look into using other X (CC look for me <3)
        x = self.last(x)
        
        return x
    
class SiameseNet_large(nn.Module):
    def __init__(self, model_choice = 1):
        super().__init__()

        if model_choice == 1:
            self.encoder = InceptionResnetV1(pretrained='vggface2').eval()
        elif model_choice == 2:
            self.encoder = InceptionResnetV1(pretrained='casia-webface').eval()

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        emb_len = 512
        self.last = nn.Sequential(
            nn.Linear(2*emb_len, 200, bias=False),
            nn.Dropout(p = 0.2), 
            nn.BatchNorm1d(200, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout(p = 0.2), 
            nn.BatchNorm1d(200, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, input1, input2):
        
        emb1 = self.encoder(input1)
        emb2 = self.encoder(input2)
        
        x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)
        x2 = torch.pow(emb1 - emb2, 2)
        #x3 = emb1 * emb2
        #x4 = emb1 + emb2
        
        x = torch.cat((x1,x2), dim=1)
        x = self.last(x)
        
        return x
    

class MultiEncoding_SiameseNet_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = InceptionResnetV1(pretrained='vggface2').eval()
        self.encoder2 = InceptionResnetV1(pretrained='casia-webface').eval()

        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        
        emb_len = 512
        self.last = nn.Sequential(
            nn.Linear(4*emb_len, 200, bias=False),
            nn.Dropout(p = 0.2), 
            nn.BatchNorm1d(200, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.Dropout(p = 0.2), 
            nn.BatchNorm1d(200, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, input1, input2):
        
        emb1 = self.encoder1(input1)
        emb2 = self.encoder1(input2)
        emb3 = self.encoder2(input1)
        emb4 = self.encoder2(input2)
        # Feature extraction2
        
        x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)
        x2 = torch.pow(emb1 - emb2, 2)
        x3 = torch.pow(emb3, 2) - torch.pow(emb4, 2)
        x4 = torch.pow(emb3 - emb4, 2)

        x = torch.cat((x1,x2, x3, x4), dim=1) # Look into using other X (CC look for me <3)
        x = self.last(x)
        
        return x
    

class MultiEncoding_SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = InceptionResnetV1(pretrained='vggface2').eval()
        self.encoder2 = InceptionResnetV1(pretrained='casia-webface').eval()

        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        
        emb_len = 512
        self.last = nn.Sequential(
            nn.Linear(4*emb_len, 200, bias=False), 
            nn.BatchNorm1d(200, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, input1, input2):
        
        emb1 = self.encoder1(input1)
        emb2 = self.encoder1(input2)
        emb3 = self.encoder2(input1)
        emb4 = self.encoder2(input2)
        # Feature extraction2
        
        x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)
        x2 = torch.pow(emb1 - emb2, 2)
        x3 = torch.pow(emb3, 2) - torch.pow(emb4, 2)
        x4 = torch.pow(emb3 - emb4, 2)

        x = torch.cat((x1,x2, x3, x4), dim=1) # Look into using other X (CC look for me <3)
        x = self.last(x)
        
        return x