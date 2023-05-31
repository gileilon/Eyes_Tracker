import os
import time
import cv2
import pygame
from TrackingPointsTask import TrackingPointsTask
from DataConvertorForTrainingClassifier import DataConvertor


class DataCreator:

    def __init__(self):
        self.task_parameters = {
            "n": 2,
            "display time": 3,
            "task time": 60*15,
            "number of frames": 3,
            "screen width": 1300,
            "screen height": 640,
            "images size": (100, 100)
        }
        self.task = TrackingPointsTask(self.task_parameters)

    def run_task_and_collect_data(self):
        counter = self.setups_for_saving_images()
        cap = cv2.VideoCapture(0)  # initializes a video capture object in OpenCV
        start_time = time.time()  # timer for display task
        # run loop for a given task time (60 = 1 minute)"
        while time.time() - start_time < self.task_parameters["task time"]:
            random_point = self.task.randomize_point()
            self.display_point_on_screen(random_point)
            frames_list = self.capture_frames_from_camera(cap, self.task_parameters["number of frames"])
            print(len(frames_list))
            for frame in frames_list:
                counter += 1
                self.crop_eyes_from_frame(frame, random_point, counter)

            time.sleep(self.task_parameters["display time"])

        cap.release()
        cv2.destroyAllWindows()
        self.task.export_data_to_excel()

    @staticmethod
    def setups_for_saving_images():
        # create "images" folder if doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')
        # setup counter in order to not overwrite files from previous runs
        num_files = len([f for f in os.listdir('images') if os.path.isfile(os.path.join('images', f))])
        if num_files != 0:
            counter = num_files  # set the counter to start from the next number after the existing files
        else:
            counter = 0
        return counter

    # Display dot on gray background in random_point location
    def display_point_on_screen(self, random_point):
        screen = pygame.display.set_mode((self.task_parameters["screen width"], self.task_parameters["screen height"]))
        screen.fill((128, 128, 128))  # gray
        pygame.draw.circle(screen, (0, 0, 0), random_point, 10)
        pygame.display.update()

    # Capture a single frame from the camera
    @staticmethod
    def capture_frames_from_camera(cap, num_frames):
        frames = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:  # Check if the capture was successful
                print('Error: Unable to capture frame from camera')
            else:
                frames.append(frame)
        return frames

    # Crop and save as imaged the eyes from input frame (using CascadeClassifier)
    def crop_eyes_from_frame(self, frame, random_point, counter):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale
        # detect the eyes in frame using CascadeClassifier:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # loop through each eye and crop the image
        eye_counter = 1
        for (x, y, w, h) in eyes:
            eye_img = frame[y:y + h, x:x + w]  # crop the image to include the eye
            eye_img = cv2.resize(eye_img, self.task_parameters["images size"])  # resize to defined output image size
            # save the cropped image and label of random point location
            filename = 'images/' + 'image_' + f'{counter}_{eye_counter}' + '.jpg'
            eye_counter += 1
            # Save the image to file
            cv2.imwrite(filename, eye_img)
            self.task.save_point_location(filename, random_point)


if __name__ == '__main__':
    data_creator = DataCreator()
    data_creator.run_task_and_collect_data()



        
    







