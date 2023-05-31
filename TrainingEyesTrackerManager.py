from DataCreator import DataCreator
from DataConvertorForTrainingClassifier import DataConvertor
from Train_SVMs import trainSVMs
import pandas as pd


class trainingEyesTrackerManager:
    def __init__(self):
        self.data_creator = DataCreator()
        self.data_convertor = DataConvertor("images")

    def create_data_from_task(self):
        self.data_creator.run_task_and_collect_data()

    def prepare_data_for_training(self):
        mSamples = self.data_convertor.convert_images_to_pixels()
        df = pd.read_csv(r"C:\Users\gilei\PycharmProjects\Eyes_Tracker\points_locations.csv")
        mLabels = df['label'].tolist()
        return mSamples, mLabels

    @staticmethod
    def train_SVMs_model(mSamples, mLabels):
        train_SVMs = trainSVMs(mSamples, mLabels)
        accuracy = train_SVMs.train_model()
        print("Accuracy:", accuracy)


if __name__ == '__main__':
    manager = trainingEyesTrackerManager()
    #manager.create_data_from_task()
    samples, labels = manager.prepare_data_for_training()
    for i in range(5):
        manager.train_SVMs_model(samples, labels)




