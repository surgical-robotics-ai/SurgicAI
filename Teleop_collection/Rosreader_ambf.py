#### Adapt to synv_record_test_ambf_v2.py
import os
import cv2
import rosbag
from glob import glob
import numpy as np

def load_images_from_directory(image_dir):
    images = []
    image_files = sorted(glob(os.path.join(image_dir, '*.png')))
    for image_file in image_files:
        img = cv2.imread(image_file)
        images.append(img)
    return images

def read_rosbag(bag_file):
    bag = rosbag.Bag(bag_file)
    topics = bag.get_type_and_topic_info()[1].keys()
    data = {topic: [] for topic in topics}
    for topic, msg, t in bag.read_messages():
        data[topic].append(msg)
    bag.close()
    return data

image_dir = "Image_data"
bag_dir = "Kinematic_data"
bag_file = os.path.join(bag_dir, 'synchronized_data.bag')

images = load_images_from_directory(image_dir)
print(f"Number of images: {len(images)}")

bag_data = read_rosbag(bag_file)
for topic, msgs in bag_data.items():
    print(f"Topic: {topic}, Number of Messages: {len(msgs)}")

cv2.destroyAllWindows()