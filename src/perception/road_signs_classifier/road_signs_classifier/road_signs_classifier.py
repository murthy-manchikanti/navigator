"""Local front-camera stop-sign detector; no inference service required."""
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from navigator_msgs.msg import RoadSigns, RoadSignsDetection
from ultralytics import YOLO
import torch
import cv2
import numpy as np


def has_stop_sign_appearance(image, box):
    """Require a filled red polygon, excluding white speed signs/red rings."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image.shape[:2]
    crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if min(crop.shape[:2], default=0) < 12:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = (((hsv[:,:,0] <= 12) | (hsv[:,:,0] >= 168)) &
           (hsv[:,:,1] >= 90) & (hsv[:,:,2] >= 55)).astype(np.uint8)*255
    ch, cw = red.shape
    center = red[ch//4:3*ch//4, cw//4:3*cw//4]
    if np.count_nonzero(red)/red.size < 0.3 or np.count_nonzero(center)/center.size < 0.3:
        return False
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 0.03*cv2.arcLength(contour, True), True)
    return 7 <= len(polygon) <= 9 and cv2.isContourConvex(polygon)


class RoadSignsClassifier(Node):
    def __init__(self):
        super().__init__('road_signs_classifier')
        self.declare_parameter('model_path', '/navigator/data/models/yolo11n.pt')
        self.declare_parameter('image_topic', '/cameras/camera0')
        self.declare_parameter('confidence', 0.6)
        model_path = self.get_parameter('model_path').value
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'Stop-sign weights missing: {model_path}')
        torch.set_num_threads(2)
        self.declare_parameter("device", "cpu")
        self.model = YOLO(model_path)
        self.sign_classes = [k for k, v in self.model.names.items() if v.lower() == 'stop sign']
        if not self.sign_classes:
            raise ValueError('Selected model has no stop sign class')
        self.bridge = CvBridge()
        self.latest = None
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.image_callback, qos_profile_sensor_data)
        self.publisher = self.create_publisher(RoadSignsDetection, '/perception/stop_signs', 10)
        self.create_timer(0.2, self.detect)

    def image_callback(self, msg):
        self.latest = msg

    def detect(self):
        msg, self.latest = self.latest, None
        if msg is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model.predict(image, classes=self.sign_classes,
                                     conf=self.get_parameter('confidence').value,
                                     device=self.get_parameter('device').value, verbose=False)
        output = RoadSignsDetection()
        output.header = msg.header
        for result in results:
            for xyxy, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                if not has_stop_sign_appearance(image, xyxy):
                    continue
                x1, y1, x2, y2 = xyxy
                sign = RoadSigns()
                sign.label = 'stop'
                sign.confidence = float(confidence)
                sign.x, sign.y = float((x1 + x2) / 2), float((y1 + y2) / 2)
                sign.width, sign.height = float(x2 - x1), float(y2 - y1)
                output.road_signs.append(sign)
        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = RoadSignsClassifier()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
