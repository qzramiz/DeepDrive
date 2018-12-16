from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset, Config
from deepgtav.client import Client
from collections import defaultdict
import gzip
import argparse
import pickle
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

class Data_Generator:

    def __init__(self, args_):
        self.__args = args_
        self.__driving_mode = [786603, 55.0]
        self.__driving_vehicle = 'Blista'
        self.__starting_time = 5
        self.__starting_weather = 4
        self.__starting_location = [-2730, 2300]
        self.__frame_size = [650, 418] 
        self.__client, self.__scenario, self.__dataset, self.__detection_pickleFile = None, None, None, None
        self.__total_written_images = 0
        self.__images_to_capture = self.__args['images_to_capture']
        self.__save_dir = self.__args['save_dir']
        self.__target_shape = (200,88)
        self.__labels_csv = None
        self.__init_client()
        self.__init_scenairo()
        self.__prepare_datasets()
        self.__client.sendMessage(Start(scenario=self.__scenario, dataset=self.__dataset))
        self.__old_location = []
        

    def __build_env(self):
        self.__init_scenairo()
        self.__prepare_datasets(open_=False)
        self.__client.sendMessage(Config(scenario=self.__scenario, dataset=self.__dataset))
        
    def __init_scenairo(self):
        self.__scenario = Scenario(drivingMode=self.__driving_mode, vehicle=self.__driving_vehicle,
                                   time=self.__starting_time, weather=self.__starting_weather, location=self.__starting_location)

    def __init_client(self):
        self.__client = Client(
            ip=self.__args['host'], port=self.__args['port'])

    def __prepare_datasets(self, open_=True, fps=20, throttle_=True, brake_=True, steering_=True, vehicles_=True, peds_=True, speed_=True, location_=True, detection_file_name_='detection1.pickle'):
        self.__dataset = Dataset(rate=fps, frame=self.__frame_size, throttle=throttle_, brake=brake_,
                          steering=steering_, vehicles=vehicles_, time=True, peds=peds_, speed=speed_, location=location_)
        if open_:
            self.__detection_pickleFile = open(detection_file_name_, mode='wb')
            self.__labels_csv = open('labels.csv', 'wb')
    
    def __save_image(self, image):
        cv2.imwrite(self.__save_dir +str(self.__total_written_images)+'.png', image)

    def __process_image(self, image, up_crop=110, down_crop=40 ):
        return cv2.resize(image[up_crop:-down_crop, :], dsize=self.__target_shape)
        

    def __recv_message(self):
        message = self.__client.recvMessage()
        image = frame2numpy(message['frame'], (self.__frame_size[0], self.__frame_size[1]))
        detect = dict()
        detect['peds'] = message['peds']
        detect['vehicles'] = message['vehicles']
        recv_labels = [message['steering'],
                           message['throttle'], message['brake'], message['speed'], message['time']]

        return message, self.__process_image(image= image), recv_labels, detect
    
    def __close_all(self):
        self.__detection_pickleFile.close()
        self.__labels_csv.close()
        self.__client.sendMessage(Stop())
        self.__client.close()

    def __dump_all(self, labels, detect_list):
        np.savetxt(self.__labels_csv, labels, delimiter=',')
        for object_ in detect_list:
            pickle.dump(object_, self.__detection_pickleFile)

    def generate_dataset(self):
        reset_images = 0
        counter = 0
        reset_threshold = 6000
        labels = np.zeros((1000,5), dtype=np.float32)
        detect_list = []

        for i in tqdm(np.arange(self.__images_to_capture - self.__total_written_images)):
            try:
                message, image, labels[counter], detect = self.__recv_message()
                self.__save_image(image)
                detect_list.append(detect)
                self.__total_written_images += 1
                reset_images += 1
                counter += 1
                if ((counter) % 1000 == 0 and counter > 0) or ((counter) % 1000 == 0 and counter > 0 and reset_images >= reset_threshold):
                    if (counter) % 1000 == 0 or reset_images >= reset_threshold:
                        #                 print ('Saved : ',total_images)
                        self.__dump_all(labels, detect_list)
                    else:
                        self.__total_written_images -= 1000
                        self.__build_env()

                    if reset_images >= reset_threshold:
                        reset_images = 0
                        self.__build_env()

                    counter = 0
                    detect_list = []

                # We send the commands predicted by the agent back to DeepGTAV to control the vehicle
            except KeyboardInterrupt:
                self.__close_all()
                break
        self.__close_all()
        print(self.__total_written_images)