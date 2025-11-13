
# https://bair.berkeley.edu/blog/2018/05/30/bdd/
# BDD100K: A Large-scale Diverse Driving Video Database
# https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k?resource=download

""" 
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
"""

from torch.utils.data import Dataset


class DetectionDataset(Dataset):

    def __init__(self, data_path, transforms=None):
        super().__init__()
        
        self.load_data(data_path)
        self.transforms = transforms
        
    def load_data(self, data_path):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass