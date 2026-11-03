"""Front-camera traffic lights mapped to Navigator's red/yellow/green labels."""
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
from navigator_msgs.msg import TrafficLight, TrafficLightDetection

# The installed model's IDs are red=0, green=1, off=2, yellow=3.
# Navigator's wire labels are red=0, yellow=1, green=2.
LABELS = {'red': 0, 'yellow': 1, 'green': 2}


class TrafficLightDetectionNode(Node):
    def __init__(self):
        super().__init__('traffic_light_detection_node')
        self.declare_parameter('model_path', '/navigator_binaries/best_traffic_med_yolo_v8.pt')
        self.declare_parameter('image_topic', '/cameras/camera0')
        self.declare_parameter('confidence', 0.6)
        model_path = self.get_parameter('model_path').value
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'Traffic-light weights missing: {model_path}')
        torch.set_num_threads(2)
        self.declare_parameter("device", "cpu")
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.latest = None
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.image_callback, qos_profile_sensor_data)
        self.traffic_light_publisher = self.create_publisher(TrafficLightDetection, '/traffic_lights/detections', 10)
        self.create_timer(0.2, self.detect)

    def image_callback(self, msg):
        self.latest = msg

    def detect(self):
        msg, self.latest = self.latest, None
        if msg is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        output = TrafficLightDetection()
        output.header = msg.header
        results = self.model.predict(image, conf=self.get_parameter('confidence').value, device=self.get_parameter('device').value, verbose=False)
        for result in results:
            for box, cls, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                label = LABELS.get(result.names[int(cls)].lower())
                if label is None:
                    continue
                x1, y1, x2, y2 = box
                light = TrafficLight()
                light.x, light.y = float(x1), float(y1)
                light.width, light.height = float(x2-x1), float(y2-y1)
                light.label, light.score = label, float(confidence)
                output.traffic_lights.append(light)
        self.traffic_light_publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
