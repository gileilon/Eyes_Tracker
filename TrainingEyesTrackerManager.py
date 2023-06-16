from DataConvertorForTrainingClassifier import DataConvertor
from Train_SVMs import trainSVMs
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from PIL import Image

class trainingEyesTrackerManager:
    def __init__(self):
        self.data_convertor = DataConvertor(str(Path(__file__).parent / Path('./data/data_2x2/images')))

    def create_data_from_task(self):
        self.data_creator.run_task_and_collect_data()

    def prepare_data_for_training_backup(self):
        mSamples = self.data_convertor.convert_images_to_pixels()
        df = pd.read_csv(str(Path(__file__).parent / Path('./data/data_2x2/points_locations.csv')))
        mLabels = df['label'].tolist()
        return mSamples, mLabels

    def prepare_data_for_training(self):
        df = pd.read_csv(str(Path(__file__).parent / Path('./data/data_2x2/points_locations.csv')))

        images = []
        labels = []
        for image_name in (Path(__file__).parent / Path('./data/data_2x2/images')).iterdir():
            if image_name.parts[-1].endswith('.jpg'):
                image = Image.open(image_name).convert('L')
                flattened_vector = np.array(image).flatten()
                images.append(flattened_vector)

                df_image_name = '/'.join(image_name.parts[-2:])
                label = df[df['filename'] == df_image_name].label.item()
                labels.append(label)

        return np.array(images), labels




    @staticmethod
    def train_SVMs_model(mSamples, mLabels):
        train_SVMs = trainSVMs(mSamples, mLabels)
        accuracy = train_SVMs.train_model()
        print("Accuracy:", accuracy)


def train_classifier(classifier, samples, labels):
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2)
    classifier.fit(x_train, y_train)
    accuracy = (np.array(y_test) == classifier.predict(x_test)).sum() / len(y_test)
    print(f'{str(classifier)}: {accuracy}')

def samples_to_components(samples):
    p = PCA(30)
    p.fit(samples)
    low_dim_samples = p.transform(samples)
    return low_dim_samples


if __name__ == '__main__':
    manager = trainingEyesTrackerManager()
    #manager.create_data_from_task()
    samples, labels = manager.prepare_data_for_training()
    samples = samples_to_components(samples)

    classifiers = [
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        MLPClassifier(hidden_layer_sizes=[256, 256, 100, 100]),
        SVC()
    ]

    for classifier in tqdm(classifiers, desc='Training classifiers'):
        train_classifier(classifier=classifier, samples=samples, labels=labels)




