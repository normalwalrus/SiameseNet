from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image

class SmileDataset(Dataset):
    def __init__(self, relations, person_to_image) -> None:
        super().__init__()

        self.relations = relations
        self.person_to_image = person_to_image
        self.people_labels = list(person_to_image.keys())
        self.image_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
            )])
    
    def __len__(self):
        #2 times since we only have positive examples and every alternate example is negative
        return len(self.relations) * 2 

    def __getitem__(self, index):

        #positive example
        if (index%2==0): 
            p1, p2 = self.relations[index//2]
            label = 1

        #negative example
        else:
            while True:
                p1 = random.choice(self.people_labels) 
                p2 = random.choice(self.people_labels) 
                if (p1 != p2) and (p1, p2) not in self.relations and (p2, p1) not in self.relations:
                    break 
            label = 0

        path1 = self.person_to_image[p1]
        path2 = self.person_to_image[p2]
        
        image1 = Image.open(path1[random.randint(0, len(path1)-1)])
        image2 = Image.open(path2[random.randint(0, len(path2)-1)])
        
        #convert_image_to_tensor_transform = transforms.ToTensor()

        image1_tensor = self.image_transform(image1)
        image2_tensor = self.image_transform(image2)

        return image1_tensor, image2_tensor, label
    

class SubmissionDataset(Dataset):
    def __init__(self, relations, person_to_image) -> None:
            super().__init__()

            self.relations = relations
            self.person_to_image = person_to_image
            self.people_labels = list(person_to_image.keys())

            self.image_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
            )])
        
    def __len__(self):
        
        return len(self.relations) 

    def __getitem__(self, index):

        p1, p2 = self.relations[index]

        path1 = self.person_to_image[p1]
        path2 = self.person_to_image[p2]
        
        image1 = Image.open(path1[0])
        image2 = Image.open(path2[0])
        
        #convert_image_to_tensor_transform = transforms.ToTensor()

        image1_tensor = self.image_transform(image1)
        image2_tensor = self.image_transform(image2)

        return image1_tensor, image2_tensor