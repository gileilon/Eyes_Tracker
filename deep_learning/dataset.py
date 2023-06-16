from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms, utils
from sklearn.utils import shuffle




class EyeDataset(Dataset):

    def _load_data(self, data_path: Path=None):
        return pd.read_csv(data_path / Path('points_locations.csv'))

    def __init__(self, data_path: Path=None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / Path('data/data_2x2')

        self.is_train = True
        self.data_path = data_path
        self.data = self._load_data(data_path=data_path)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ])

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(self.data_path / Path(row['filename']))
        transform = self.train_transform if self.is_train else self.test_transform
        return {'image': transform(img), 'label': row['label']}

    def get_number_of_labels(self):
        return len(self.data['label'].unique())


def random_split(dataset, test_percentage: float):
        all_data = dataset.data
        all_data = shuffle(all_data)
        train, test = EyeDataset(), EyeDataset()
        test_index = int(test_percentage * len(all_data))
        test.data = all_data.iloc[:test_index]
        train.data = all_data.iloc[test_index:]
        return train, test


def get_dataloaders(test_percentage=0.2, batch_size=4):
    full_data = EyeDataset()
    train_dataset, test_dataset = random_split(full_data, test_percentage=test_percentage)
    test_dataset.is_train = False
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False), full_data.get_number_of_labels()


if __name__ == '__main__':
    d = EyeDataset()
    a = d[100]

