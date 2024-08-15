import os
import rospy
from sensor_msgs.msg import Image, JointState
from ambf_msgs.msg import RigidBodyState
from message_filters import ApproximateTimeSynchronizer, Subscriber
import threading
import cv2
from cv_bridge import CvBridge
import rosbag
import time
import queue

# Global image queue
image_queue = queue.Queue()

# Global reference to the synchronizer
synchronizer = None

def image_saver(queue):
    global synchronizer
    while True:
        item = queue.get()
        if item is None:
            break
        filename, image, is_visual = item
        if is_visual:
            cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, 85])  # JPEG with compression for visualized images
        else:
            cv2.imwrite(filename, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # PNG without compression for main images
        synchronizer.images_saved += 1
        queue.task_done()

class ImageSaver:
    def __init__(self,topic_name) -> None:
        self.topic_name = topic_name
        

class TopicSynchronizer:
    def __init__(self, topic_types, is_visualized=False):
        self.topic_types = topic_types
        self.topic_names = list(topic_types.keys())
        self.bridge = CvBridge()
        self.non_image_data = []  
        self.subscribers = []
        self.is_visualized = is_visualized

        # Initialize the ROS node
        rospy.init_node('dynamic_topic_sync_node')

        # Create subscribers for each topic with its corresponding type
        for topic, msg_type in self.topic_types.items():
            rospy.loginfo(f"Subscribing to topic: {topic} with type: {msg_type}")
            subscriber = Subscriber(topic, msg_type)
            self.subscribers.append(subscriber)

        # ApproximateTimeSynchronizer to synchronize the topics
        self.ats = ApproximateTimeSynchronizer(self.subscribers, queue_size=100, slop=0.1)
        self.ats.registerCallback(self.synchronized_callback)

        # Ensure the directory for saving images and rosbags exists
        self.image_dir = os.path.join(os.getcwd(), "Image_data")
        self.visual_dir = os.path.join(os.getcwd(), "Visual_image_data")
        self.bag_dir = os.path.join(os.getcwd(), "Kinematic_data")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.visual_dir, exist_ok=True)
        os.makedirs(self.bag_dir, exist_ok=True)

        self.image_count = 0
        self.stop_flag = False
        self.images_received = 0
        self.images_saved = 0

    def synchronized_callback(self, *args):
        if self.stop_flag:
            return

        for topic_name, msg in zip(self.topic_names, args):
            if isinstance(msg, Image):
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                filename = os.path.join(self.image_dir, f"image_{self.image_count:06d}.png")
                image_queue.put((filename, cv_image, False))  # False indicates it's not a visual image
                
                if self.is_visualized:
                    visual_filename = os.path.join(self.visual_dir, f"image_{self.image_count:06d}.jpg")
                    image_queue.put((visual_filename, cv_image, True))  # True indicates it's a visual image
                
                self.image_count += 1
                self.images_received += 1
            else:
                self.non_image_data.append((topic_name, msg, msg.header.stamp))

    def save_kinematic_data(self):
        # Write all kinematics data to a rosbag
        bag_path = os.path.join(self.bag_dir, 'synchronized_data.bag')
        with rosbag.Bag(bag_path, 'w') as bag:
            for topic_name, msg, stamp in self.non_image_data:
                bag.write(topic_name, msg, stamp)

    def start(self):
        self.start_time = time.time()
        rospy.spin()

    def stop_and_report(self):
        self.stop_flag = True
        
        # Wait for all images to be saved
        while self.images_saved < self.images_received:
            time.sleep(0.1)
        
        elapsed_time = time.time() - self.start_time
        print(f"Total time: {elapsed_time:.3f} seconds")
        print(f"Total images collected: {self.image_count}")
        print(f"Average collection frequency: {self.image_count / elapsed_time:.2f} Hz")
        
        # Signal the worker thread to stop
        image_queue.put(None)
        worker.join()

if __name__ == '__main__':
    topic_types = {
        '/ambf/env/cameras/cameraL/ImageData': Image,
        '/ambf/env/Needle/State': RigidBodyState,
        '/ambf/env/psm1/baselink/State': RigidBodyState,
        '/ambf/env/psm2/baselink/State': RigidBodyState,
        '/MTML/gripper/measured_js': JointState,
        '/MTMR/gripper/measured_js': JointState

    }

    # is_visualized=True to save both PNG and JPEG (visual) images
    synchronizer = TopicSynchronizer(topic_types, is_visualized=True)

    # Start worker thread
    worker = threading.Thread(target=image_saver, args=(image_queue,))
    worker.start()

    spin_thread = threading.Thread(target=synchronizer.start)
    spin_thread.start()

    input("Press Enter to stop and save all data...")
    synchronizer.save_kinematic_data()
    synchronizer.stop_and_report()
    rospy.signal_shutdown('Data saved and shutdown initiated')