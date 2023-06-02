import torch
import pickle
import Config
import numpy as np
from torchvision import transforms
from SiameseNet import SiameseNet

class SignatureVerifier:
    def __init__(self, weights_path, svm_model_path, device='cuda'):

        self.device = device

        self.model = SiameseNet(Config.ARCHITECTURE_CFG)
        self.model.load_state_dict(torch.load(weights_path)['model'])
        self.model.eval()
        self.model = self.model.to(device)

        self.transforms = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE,Config.IMG_SIZE)),
            transforms.ToTensor()
            ])
        
        self.svm = pickle.load(open(svm_model_path, 'rb'))

    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    @staticmethod
    def manhattan_distance(a, b):
        return np.sum(np.abs(a - b))

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def create_embeds(self, image1, image2, transform=True):
        if transform:
            image1 = self.transforms(image1).unsqueeze(0)
            image2 = self.transforms(image2).unsqueeze(0)
        
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        embed1, embed2 = self.model(image1, image2)

        return embed1.detach().cpu().numpy().squeeze(), embed2.detach().cpu().numpy().squeeze()
    
    def __call__(self, image1, image2):
        embed1, embed2 = self.create_embeds(image1, image2)
        e_dist = self.euclidean_distance(embed1, embed2)
        m_dist = self.manhattan_distance(embed1, embed2)
        c_sim = self.cosine_similarity(embed1, embed2)
        pred = self.svm.predict(np.array([e_dist, m_dist, c_sim]))
        return pred
