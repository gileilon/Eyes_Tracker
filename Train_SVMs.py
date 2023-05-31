from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class trainSVMs:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels


    def train_model(self):
        # Step 1: Split the data into training and testing sets
        x_train, x_test, y_train, y_test = self.split_data()

        # Step 2: Create an instance of the SVM model and fit it to the training data
        model = SVC()
        model.fit(x_train, y_train)

        # Step 3: Predict labels for the test set
        y_predicted = model.predict(x_test)

        # Step 4: Evaluate the performance of the model
        return self.evaluate_performance(y_test, y_predicted)

    def split_data(self):
        x_train, x_test, y_train, y_test = train_test_split(self.samples, self.labels, test_size=0.2)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def evaluate_performance(y_test, y_predicted):
        accuracy = accuracy_score(y_test, y_predicted)
        return accuracy
        # print("Accuracy:", accuracy)
