import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = ("*.jpg","*.jpeg","*.JPG","*.png")

# This file contains the dataset for the PlantVillage dataset; see README.md for more details

class PlantDataset(Dataset):
    """
    root_dir/
        Tomato___Early_blight/
            img_001.jpg
            img_002.jpg
        Tomato___healthy/
            ...
        Potato___Late_blight/
            ...
        Background_without_leaves/
            ...
    """
    def __init__(self, root_dir, transform=None, res=(256, 256)):
        self.root_dir   = Path(root_dir)
        self.transform  = transform or T.ToTensor()
        self.res        = res

        self.samples          = []   # [image_path, plant_label, disease_label]
        plants, diseases      = set(), set()

        # Traverse the dataset directory structure
        for sub in self.root_dir.iterdir():
            if not sub.is_dir(): 
                continue
            name = sub.name
            if name == "Background_without_leaves":
                label = ("Background", "None")
            else:
                try:
                    plant, disease = name.split("___", 1)   # split only on the first triple‑_
                except ValueError:
                    raise ValueError(f"Bad folder name: {name}")
                label = plant.replace("_", " "), disease.replace("_", " ")
            for pattern in IMG_EXTS:
                for img_file in sub.glob(pattern):
                    self.samples.append((img_file, *label))
                    plants.add(label[0])
                    diseases.add(label[1])

        # Dictionary in form {plant: index}
        self.plant_to_idx   = {plant: i for i, plant in enumerate(sorted(plants))}
        self.disease_to_idx = {disease: i for i, disease in enumerate(sorted(diseases))}

        # Targets is a list of tuples (plant_idx, disease_idx)
        self.targets = [
            (self.plant_to_idx[plant],
             self.disease_to_idx[disease])
            for _, plant, disease in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, _, _ = self.samples[index]
        plant_idx, disease_idx = self.targets[index]

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
            img = T.Resize(self.res)(img)
            img = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

        return img, (plant_idx, disease_idx)