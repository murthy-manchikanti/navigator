"""Regression tests for the local planner / ACC contract (ROS Humble)."""
import copy
import math
from types import SimpleNamespace
import pytest
import rclpy
from rclpy.time import Time
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from rosgraph_msgs.msg import Clock
from path_planners.path_planner_node import PathPlannerNode
from autonomous_cruise.autonomous_cruise_node import AutonomousCruiseController
from intersection_manager.intersection_manager import IntersectionManager
from costs.route_costmap_node import RouteCostmapNode

@pytest.fixture
def pipeline():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true', '-p', 'traffic_rule_source:=camera'])
    planner = PathPlannerNode()
    controller = AutonomousCruiseController()
    intersection = IntersectionManager()
    route = RouteCostmapNode()
    nodes = [planner, controller, intersection, route]
    for node in nodes:
        node.get_clock().set_ros_time_override(Time(seconds=10))
    clock = Clock(); clock.clock.sec = 10
    planner.clock_callback(clock); route.clockCb(clock)
    controller.current_odom = Odometry()
    controls = []; paths = []
    controller.control_pub = SimpleNamespace(publish=lambda m: controls.append(copy.deepcopy(m)))
    def receive(path):
        paths.append(copy.deepcopy(path))
        controller.path_callback(path)
        intersection.routeCallback(path)
    planner.path_pub = SimpleNamespace(publish=receive)
    grid = OccupancyGrid(); grid.header.frame_id = 'base_link'; grid.header.stamp = copy.deepcopy(clock.clock)
    grid.info.resolution = 0.2; grid.info.height = 60; grid.info.width = 100
    grid.info.origin.position.x = -4.0; grid.info.origin.position.y = -6.0
    grid.data = [0] * 6000
    planner.costmap_callback(grid)
    goal = PoseStamped(); goal.header.frame_id='base_link'; goal.header.stamp=copy.deepcopy(clock.clock)
    goal.pose.position.x = 6.0
    planner.path_goal_callback(goal)
    yield SimpleNamespace(planner=planner, controller=controller, intersection=intersection,
                          route=route, grid=grid, goal=goal, controls=controls, paths=paths)
    for node in nodes: node.destroy_node()
    rclpy.shutdown()

@pytest.mark.parametrize('y,sign', [(2.0,-1),(-2.0,1),(0.0,0)])
def test_rectangular_grid_to_acc(pipeline, y, sign):
    p=pipeline; p.goal.pose.position.y=y
    p.planner.generate_path(); p.controller.control_loop()
    path=p.paths[-1]
    assert len(path.poses)>10
    assert abs(path.poses[0].pose.position.x)<0.21
    assert abs(path.poses[0].pose.position.y)<0.21
    assert path.poses[-1].pose.position.x==pytest.approx(6.0)
    assert path.poses[-1].pose.position.y==pytest.approx(y)
    if sign: assert p.controls[-1].steer*sign>0
    else: assert abs(p.controls[-1].steer)<1e-5

@pytest.mark.parametrize('failure', ['blocked','missing_goal','missing_grid','stale_goal','stale_grid','wrong_frame','bad_size'])
def test_planning_failure_brakes(pipeline, failure):
    p=pipeline; p.planner.generate_path(); assert p.paths[-1].poses
    if failure=='blocked': p.grid.data=[100]*6000
    elif failure=='missing_goal': p.planner.path_goal=None
    elif failure=='missing_grid': p.planner.costmap=None
    elif failure=='stale_goal': p.goal.header.stamp.sec=1
    elif failure=='stale_grid': p.grid.header.stamp.sec=1
    elif failure=='wrong_frame': p.grid.header.frame_id='map'
    elif failure=='bad_size': p.grid.data=[]
    p.planner.generate_path(); p.controller.control_loop()
    assert not p.paths[-1].poses
    assert p.controls[-1].throttle==0 and p.controls[-1].brake==1

@pytest.mark.parametrize('expiry', ['silence','old_stamp','future_stamp','wrong_frame','clock_reset'])
def test_acc_expired_path_brakes(pipeline, expiry):
    p=pipeline; p.planner.generate_path()
    if expiry=='silence': p.controller.get_clock().set_ros_time_override(Time(seconds=11))
    elif expiry=='old_stamp': p.controller.current_path.header.stamp.sec=1
    elif expiry=='future_stamp': p.controller.current_path.header.stamp.sec=100
    elif expiry=='wrong_frame': p.controller.current_path.header.frame_id='map'
    elif expiry=='clock_reset': p.controller.get_clock().set_ros_time_override(Time(seconds=1))
    p.controller.control_loop()
    assert p.controls[-1].throttle==0 and p.controls[-1].brake==1

@pytest.mark.parametrize('n',[0,1,10,11,45,109,110,150])
def test_intersection_short_paths(pipeline,n):
    path=Path(); path.poses=[PoseStamped() for _ in range(n)]
    pipeline.intersection.behavior_through_intersection='LeftTurn'
    pipeline.intersection.routeCallback(path)
    if n<=10: assert pipeline.intersection.behavior_through_intersection is None


@pytest.mark.parametrize("camera_confirmed", [False, True])
def test_route_repaints_for_yaw_and_small_translation(pipeline, camera_confirmed):
    p=pipeline; node=p.route
    node._cfg={'occupancy_grids':{'resolution':0.2,'width':60.,'length':60.,
             'vehicle_longitudinal_location':20.,'vehicle_latitudinal_location':30.}}
    route=Path()
    for x in range(21):
        pose=PoseStamped(); pose.pose.position.x=float(x); route.poses.append(pose)
    node.routeCb(route)
    if camera_confirmed:
        import numpy as np
        node._drivable_grid = np.zeros((300, 300), dtype=np.int8)
    tf=TransformStamped();tf.transform.rotation.w=1.
    node.tf_buffer=SimpleNamespace(lookup_transform=lambda *a,**kw:tf)
    results=[]; node.publish=lambda grid,goal,cfg:results.append((grid.copy(),goal))
    node.buildRouteCostmap(); first=results[-1]
    tf.transform.translation.y=0.2
    node.buildRouteCostmap(); second=results[-1]
    assert first[1]!=second[1]
    assert math.hypot(*second[1])<=8.01
    tf.transform.rotation.z=math.sin(0.1/2);tf.transform.rotation.w=math.cos(0.1/2)
    node.buildRouteCostmap();third=results[-1]
    assert second[1]!=third[1]
    assert not (second[0]==third[0]).all()


def test_route_end_publishes_without_exception(pipeline):
    node=pipeline.route
    node._cfg={'occupancy_grids':{'resolution':0.2,'width':60.,'length':60.,
             'vehicle_longitudinal_location':20.,'vehicle_latitudinal_location':30.}}
    route=Path();pose=PoseStamped();pose.pose.position.x=1.;route.poses=[pose]
    node.routeCb(route)
    tf=TransformStamped();tf.transform.rotation.w=1.
    node.tf_buffer=SimpleNamespace(lookup_transform=lambda *a,**kw:tf)
    goals=[];node.publish=lambda grid,goal,cfg:goals.append(goal)
    node.buildRouteCostmap();assert goals[-1]==(0.,0.)


def test_terminal_goal_brakes(pipeline):
    p=pipeline; p.goal.pose.position.x=0.0
    p.planner.generate_path(); p.controller.control_loop()
    assert len(p.paths[-1].poses)==1
    assert p.controls[-1].throttle==0 and p.controls[-1].brake==1


def test_new_goal_after_clock_reset(pipeline):
    p=pipeline; clock=Clock(); clock.clock.sec=1
    p.planner.clock_callback(clock)
    p.goal.header.stamp.sec=1
    p.planner.path_goal_callback(p.goal)
    assert p.planner.path_goal is p.goal


def lidar_cloud(distance, count=8, y=0.0):
    from sensor_msgs.msg import PointCloud2, PointField
    import struct
    cloud = PointCloud2()
    cloud.height = 1
    cloud.width = count
    cloud.point_step = 12
    cloud.row_step = count * 12
    cloud.fields = [PointField(name=name, offset=i * 4, datatype=7, count=1)
                    for i, name in enumerate(('x', 'y', 'z'))]
    cloud.data = b''.join(struct.pack('<fff', distance, y, 0.8) for _ in range(count))
    return cloud


def acc_tick(pipeline, seconds):
    controller = pipeline.controller
    controller.get_clock().set_ros_time_override(Time(seconds=seconds))
    path = copy.deepcopy(pipeline.paths[-1])
    path.header.stamp = controller.get_clock().now().to_msg()
    controller.path_callback(path)
    controller.control_loop()
    return pipeline.controls[-1]


@pytest.mark.parametrize('dropout', ['empty', 'sparse', 'nan', 'malformed', 'path_shift', 'threshold_noise'])
def test_obstacle_stop_survives_detection_loss(pipeline, dropout):
    p = pipeline
    p.planner.generate_path()
    p.controller.longitudinal_controller.integral = 5.0
    p.controller.longitudinal_controller.last_a = 2.5
    p.controller.lidar_callback(lidar_cloud(1.8))
    assert acc_tick(p, 10.0).brake == 1.0
    for i in range(1, 31):
        p.controller.get_clock().set_ros_time_override(Time(seconds=10.0 + i * 0.1))
        cloud = lidar_cloud(1.8)
        if dropout == 'empty': cloud = lidar_cloud(1.8, count=0)
        elif dropout == 'sparse': cloud = lidar_cloud(1.8, count=3)
        elif dropout == 'nan': cloud = lidar_cloud(float('nan'))
        elif dropout == 'malformed': cloud.fields = []
        elif dropout == 'threshold_noise': cloud = lidar_cloud(2.1)
        elif dropout == 'path_shift':
            for pose in p.controller.current_path.poses: pose.pose.position.y = 4.0
        p.controller.lidar_callback(cloud)
        control = acc_tick(p, 10.0 + i * 0.1)
        assert control.throttle == 0.0 and control.brake == 1.0
    assert p.controller.obstacle_stop_latched
    assert p.controller.longitudinal_controller.integral == 0.0


def test_obstacle_stop_releases_after_confirmed_clearance(pipeline):
    p = pipeline
    p.planner.generate_path()
    p.controller.lidar_callback(lidar_cloud(1.8))
    acc_tick(p, 10.0)
    for i in range(1, 10):
        p.controller.get_clock().set_ros_time_override(Time(seconds=10.0 + i * 0.1))
        p.controller.lidar_callback(lidar_cloud(10.0))
        assert acc_tick(p, 10.0 + i * 0.1).brake == 1.0
    for i in range(10, 15):
        p.controller.get_clock().set_ros_time_override(Time(seconds=10.0 + i * 0.1))
        p.controller.lidar_callback(lidar_cloud(10.0))
        acc_tick(p, 10.0 + i * 0.1)
    assert not p.controller.obstacle_stop_latched
    assert p.controls[-1].throttle > 0.0


def test_clearance_confirmation_does_not_bridge_sensor_gap(pipeline):
    p = pipeline
    p.planner.generate_path()
    p.controller.lidar_callback(lidar_cloud(1.8))
    acc_tick(p, 10.0)
    p.controller.lidar_callback(lidar_cloud(10.0))
    p.controller.get_clock().set_ros_time_override(Time(seconds=12.0))
    p.controller.lidar_callback(lidar_cloud(10.0))
    assert acc_tick(p, 12.0).brake == 1.0
    assert p.controller.obstacle_stop_latched


def test_near_field_obstacle_holds_brake(pipeline):
    p = pipeline
    p.planner.generate_path()
    p.controller.lidar_callback(lidar_cloud(0.8))
    control = acc_tick(p, 10.0)
    assert control.throttle == 0.0 and control.brake == 1.0



def test_stopped_object_brakes_despite_pid_windup(pipeline):
    from navigator_msgs.msg import Object3D, Object3DArray
    controller = pipeline.controller.longitudinal_controller
    controller.integral = 5.0
    controller.last_a = 2.5
    objects = Object3DArray()
    car = Object3D()
    car.label = Object3D.CAR
    car.bounding_box.coordinates = [3.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    objects.objects = [car]
    throttle, brake, target = controller.compute_control(0.0, objects, 0.05)
    assert throttle == 0.0 and brake == 1.0 and target == 0.0
    assert controller.integral == 0.0 and controller.last_a == 0.0


def test_stop_sign_full_stop_and_dwell(pipeline):
    from autonomous_cruise.traffic_rules import StopSignPolicy
    policy = StopSignPolicy()
    policy.observe(3.0, 10.0)
    assert policy.limit(0.0, 10.0) == (None, False)
    policy.observe(3.0, 10.1)
    assert policy.limit(0.0, 10.1) == (0.0, True)
    assert policy.limit(0.0, 12.9) == (0.0, True)
    assert policy.limit(0.0, 13.2) == (None, False)
    policy.observe(3.0, 13.3)
    assert policy.limit(0.0, 13.3) == (None, False)


def test_stop_sign_approach_and_detection_dropout(pipeline):
    from autonomous_cruise.traffic_rules import StopSignPolicy
    policy = StopSignPolicy()
    policy.observe(8.0, 10.0); policy.observe(8.0, 10.1)
    limit, brake = policy.limit(2.0, 10.1)
    assert 0 < limit < 4 and not brake
    for i in range(1, 26):
        limit, brake = policy.limit(2.0, 10.1 + i * 0.1)
    assert brake
    assert policy.limit(1.0, 20.0)[1]
    assert policy.stopped_since is None


def test_stop_sign_rearm_requires_distance_and_absence(pipeline):
    from autonomous_cruise.traffic_rules import StopSignPolicy
    policy = StopSignPolicy(hold_seconds=0)
    policy.observe(3.0, 1.0); policy.observe(3.0, 1.1)
    policy.limit(0.0, 1.1)
    assert policy.served
    for i in range(1, 61): policy.limit(2.0, 1.1 + i * 0.1)
    assert not policy.served
    policy.observe(3.0, 8.0); policy.observe(3.0, 8.1)
    assert policy.limit(1.0, 8.1)[1]


def test_camera_lights_and_simulator_override(pipeline):
    from navigator_msgs.msg import TrafficLight, TrafficLightDetection
    from std_msgs.msg import String
    p=pipeline; p.planner.generate_path()
    msg=TrafficLightDetection(); msg.header.stamp.sec=10
    red=TrafficLight(); red.label=0; red.score=0.9; red.height=20.0
    msg.traffic_lights=[red]
    p.controller.camera_light_callback(msg)
    assert acc_tick(p, 10.0).brake == 1.0
    msg.traffic_lights=[]; p.controller.camera_light_callback(msg)
    assert acc_tick(p, 10.0).brake == 1.0
    green=TrafficLight(); green.label=2; green.score=0.9; green.height=20.0
    msg.traffic_lights=[green]; p.controller.camera_light_callback(msg)
    assert not p.controller.camera_light_stop
    p.controller.traffic_light_callback(String(data='Yellow'))
    assert acc_tick(p, 10.0).brake == 1.0


def test_stop_sign_detection_to_acc_brake(pipeline):
    from navigator_msgs.msg import RoadSigns, RoadSignsDetection
    p=pipeline; p.planner.generate_path()
    msg=RoadSignsDetection(); msg.header.stamp.sec=10
    sign=RoadSigns(); sign.label='stop'; sign.confidence=0.9; sign.height=160.0
    msg.road_signs=[sign]
    p.controller.stop_sign_callback(msg)
    assert p.controller.stop_sign_policy.distance is None
    p.controller.stop_sign_callback(msg)
    assert acc_tick(p, 10.0).brake == 1.0
    assert not p.controller.stop_sign_policy.served


def test_empty_light_callback_does_not_crash_manager(pipeline):
    from navigator_msgs.msg import TrafficLightDetection, TrafficLight
    p=pipeline
    msg=TrafficLightDetection()
    p.intersection.trafficLightCallback(msg)
    light=TrafficLight(); light.label=0; light.score=0.9
    msg.traffic_lights=[light]
    p.intersection.trafficLightCallback(msg)
    assert p.intersection.classify_intersection_type() == 'TrafficLight'
    assert p.intersection.navigate_traffic_light_intersection() == 'BrakeAndStop'
    light.label=2
    p.intersection.trafficLightCallback(msg)
    assert p.intersection.navigate_traffic_light_intersection() == 'KeepDriving'


def test_light_model_labels_translate_to_wire_contract(pipeline, monkeypatch):
    import numpy as np
    from cv_bridge import CvBridge
    from traffic_light_detector import traffic_light_node as module
    class Values:
        def __init__(self, value): self.value=value
        def tolist(self): return self.value
    boxes=SimpleNamespace(xyxy=Values([[1, 2, 10, 20]]*4),
                          cls=Values([0,1,2,3]), conf=Values([0.9]*4))
    result=SimpleNamespace(names={0:'red',1:'green',2:'off',3:'yellow'},boxes=boxes)
    monkeypatch.setattr(module.Path, 'is_file', lambda self: True)
    monkeypatch.setattr(module, 'YOLO', lambda path: SimpleNamespace(predict=lambda *a,**k:[result]))
    node=module.TrafficLightDetectionNode()
    try:
        messages=[];node.traffic_light_publisher=SimpleNamespace(publish=messages.append)
        image=CvBridge().cv2_to_imgmsg(np.zeros((32,32,3),dtype=np.uint8),'bgr8')
        image.header.stamp.sec=10
        node.image_callback(image);node.detect();node.detect()
        assert len(messages)==1
        assert [light.label for light in messages[0].traffic_lights]==[0,2,1]
        assert all(light.score > 0.8 for light in messages[0].traffic_lights)
        assert messages[0].header.stamp.sec==10
    finally:
        node.destroy_node()


def test_stop_detector_publishes_all_signs_and_empty_frames(pipeline, monkeypatch):
    import numpy as np
    from cv_bridge import CvBridge
    from road_signs_classifier import road_signs_classifier as module
    class Values:
        def __init__(self, value): self.value=value
        def tolist(self): return self.value
    boxes=SimpleNamespace(xyxy=Values([[1,2,10,20],[12,3,22,23]]),conf=Values([0.9,0.95]))
    def predict(*args, **kwargs):
        assert kwargs['classes']==[11]
        return [SimpleNamespace(boxes=boxes)]
    monkeypatch.setattr(module.Path, 'is_file', lambda self: True)
    monkeypatch.setattr(module, 'YOLO', lambda path: SimpleNamespace(names={11:'stop sign'},predict=predict))
    monkeypatch.setattr(module, 'has_stop_sign_appearance', lambda image, box: True)
    node=module.RoadSignsClassifier()
    try:
        messages=[];node.publisher=SimpleNamespace(publish=messages.append)
        image=CvBridge().cv2_to_imgmsg(np.zeros((32,32,3),dtype=np.uint8),'bgr8')
        image.header.stamp.sec=10
        node.image_callback(image);node.detect()
        assert len(messages[0].road_signs)==2
        assert all(sign.label=='stop' for sign in messages[0].road_signs)
        boxes.xyxy=Values([]);boxes.conf=Values([])
        node.image_callback(image);node.detect()
        assert messages[-1].road_signs==[]
    finally:
        node.destroy_node()


def test_stale_light_cannot_release_red(pipeline):
    from navigator_msgs.msg import TrafficLight, TrafficLightDetection
    controller=pipeline.controller
    controller.camera_light_stop=True
    msg=TrafficLightDetection();msg.header.stamp.sec=1
    light=TrafficLight();light.label=2;light.score=0.9;light.height=20.0
    msg.traffic_lights=[light]
    controller.camera_light_callback(msg)
    assert controller.camera_light_stop


def set_scene(pipeline, **facts):
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
    import json
    controller=pipeline.controller
    controller.set_parameters([Parameter('traffic_rule_source', value='carla')])
    data={'valid':True,'stamp':10.0,'vehicle_gap':None,'ego_front':2.5,
          'stop_distance':None,'stop_id':None,'light_state':'None','light_distance':None}
    data.update(facts)
    controller.scene_callback(String(data=json.dumps(data)))


def test_carla_speed_sign_false_positive_cannot_stop(pipeline):
    from navigator_msgs.msg import RoadSigns, RoadSignsDetection
    p=pipeline;set_scene(p)
    msg=RoadSignsDetection();msg.header.stamp.sec=10
    sign=RoadSigns();sign.label='stop';sign.confidence=0.99;sign.height=200.0
    msg.road_signs=[sign]
    p.controller.stop_sign_callback(msg);p.controller.stop_sign_callback(msg)
    assert p.controller.stop_sign_policy.distance is None


@pytest.mark.parametrize('gap',[0.0,1.0,1.5])
def test_api_vehicle_clearance_holds_brake(pipeline,gap):
    p=pipeline;p.planner.generate_path();set_scene(p,vehicle_gap=gap)
    assert acc_tick(p,10.0).brake==1.0
    for i in range(1,16):
        stamp=10.0+i*0.1
        p.controller.get_clock().set_ros_time_override(Time(seconds=stamp))
        set_scene(p,vehicle_gap=gap,stamp=stamp)
        p.controller.lidar_callback(lidar_cloud(10.0))
        control=acc_tick(p,stamp)
        assert control.throttle==0.0 and control.brake==1.0


def test_pre_stop_lidar_dropout_does_not_restore_cruise(pipeline):
    p=pipeline;p.planner.generate_path();p.controller.current_speed=2.0
    p.controller.lidar_callback(lidar_cloud(6.0))
    acc_tick(p,10.0)
    p.controller.lidar_callback(lidar_cloud(6.0,count=0))
    acc_tick(p,10.1)
    assert p.controller.longitudinal_controller.target_speed < p.controller.target_speed
    p.controller.current_speed=0.0
    control=acc_tick(p,10.2)
    assert control.throttle==0 and control.brake==1


def test_api_vehicle_vetoes_lidar_latch_release(pipeline):
    p=pipeline;p.planner.generate_path();set_scene(p,vehicle_gap=1.0)
    p.controller.lidar_callback(lidar_cloud(4.0))
    assert p.controller.obstacle_stop_latched
    for i in range(1,16):
        stamp=10.0+i*0.1
        p.controller.get_clock().set_ros_time_override(Time(seconds=stamp))
        set_scene(p,stamp=stamp,vehicle_gap=1.0)
        p.controller.lidar_callback(lidar_cloud(12.0))
    assert p.controller.obstacle_stop_latched


def test_api_missing_or_stale_holds_brake(pipeline):
    p=pipeline;p.planner.generate_path();set_scene(p,stamp=1.0)
    assert acc_tick(p,10.0).brake==1.0
    set_scene(p,valid=False)
    assert acc_tick(p,10.0).brake==1.0


@pytest.mark.parametrize('state',['Red','Yellow'])
def test_api_light_brakes_at_lane_stop_line(pipeline,state):
    p=pipeline;p.planner.generate_path()
    set_scene(p,light_state=state,light_distance=4.5)
    assert acc_tick(p,10.0).brake==1.0


def test_appearance_rejects_round_speed_sign(pipeline):
    import cv2
    import numpy as np
    from road_signs_classifier.road_signs_classifier import has_stop_sign_appearance
    image=np.zeros((120,120,3),dtype=np.uint8)
    cv2.circle(image,(60,60),48,(0,0,255),-1)
    cv2.circle(image,(60,60),39,(255,255,255),-1)
    cv2.putText(image,'30',(35,72),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    assert not has_stop_sign_appearance(image,[10,10,110,110])
    image[:]=0
    points=np.array([[40,12],[80,12],[108,40],[108,80],[80,108],[40,108],[12,80],[12,40]])
    cv2.fillPoly(image,[points],(0,0,255))
    cv2.putText(image,'STOP',(23,66),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    assert has_stop_sign_appearance(image,[10,10,110,110])


@pytest.mark.parametrize('layout',['horizontal','vertical'])
def test_api_scene_uses_lane_light_state_independent_of_layout(pipeline,layout):
    import importlib.util
    import fnmatch
    from pathlib import Path as FilePath
    observer=FilePath(__file__).resolve().parents[2]/'docker/carla-ros-bridge/carla_scene_observer.py'
    spec=importlib.util.spec_from_file_location('scene_observer_test',observer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def location(x=0,y=0,z=0):return SimpleNamespace(x=x,y=y,z=z)
    class Transform:
        def __init__(self,x=0,y=0):self.location=location(x,y);self.rotation=SimpleNamespace(yaw=0)
        def get_forward_vector(self):return location(1,0)
        def transform(self,p):return location(self.location.x+p.x,self.location.y+p.y,p.z)
        def get_inverse_matrix(self):return [[1,0,0,-self.location.x],[0,1,0,-self.location.y],[0,0,1,0],[0,0,0,1]]
    class Waypoint:
        road_id=1;lane_id=1
        def __init__(self,x):self.s=x;self.transform=Transform(x)
        def next(self,d):return [Waypoint(self.s+d)] if self.s<30 else []
    def vehicle(actor_id,x):
        tf=Transform(x)
        return SimpleNamespace(id=actor_id,type_id='vehicle.test',get_transform=lambda:tf,
            get_traffic_light=lambda:None,bounding_box=SimpleNamespace(location=location(),extent=location(2.5,1,1)))
    def sign(actor_id,kind,y):
        return SimpleNamespace(id=actor_id,type_id=kind,get_transform=lambda:Transform(10,y),
            trigger_volume=SimpleNamespace(location=location(),rotation=SimpleNamespace(yaw=0),extent=location(1,1,1)))
    ego=vehicle(1,0);lead=vehicle(2,8)
    light=SimpleNamespace(id=8,type_id='traffic.traffic_light',layout=layout,
        get_stop_waypoints=lambda:[Waypoint(12)],get_state=lambda:'Red')
    class Actors(list):
        def filter(self,pattern):return [a for a in self if fnmatch.fnmatch(a.type_id,pattern)]
    actors=Actors([ego,lead,light,sign(4,'traffic.speed_limit.30',0),sign(5,'traffic.stop',8)])
    world=SimpleNamespace(get_actors=lambda:actors,get_map=lambda:SimpleNamespace(get_waypoint=lambda loc:Waypoint(0)),
        get_traffic_lights_from_waypoint=lambda wp,d:[light],
        get_snapshot=lambda:SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=10)))
    result=module.scene(world,ego)
    assert result['stop_id'] is None  # speed-limit and cross-street signs excluded
    assert result['light_state']=='Red'
    assert 10 < result['light_distance'] < 13
    assert result['vehicle_gap']==pytest.approx(3.0)  # bumper-to-bumper, not center distance
    actors.append(sign(6,'traffic.stop',0))
    result=module.scene(world,ego)
    assert result['stop_id']==6
    assert 8 <= result['stop_distance'] <= 10


@pytest.mark.parametrize('gap,stopped', [(1.4, True), (1.5, True), (1.8, False), (2.7, False)])
def test_tuned_standstill_gap(pipeline, gap, stopped):
    p=pipeline;p.planner.generate_path();set_scene(p,vehicle_gap=gap,ego_front=2.4)
    assert p.controller._lidar_stop_distance()==pytest.approx(3.9)
    control=acc_tick(p,10.0)
    if stopped:
        assert control.brake==1 and control.throttle==0
    else:
        assert control.throttle>0 and control.brake==0


def test_lidar_and_api_use_same_standstill_gap(pipeline):
    p=pipeline;p.planner.generate_path();set_scene(p,vehicle_gap=1.8,ego_front=2.4)
    p.controller.lidar_callback(lidar_cloud(4.2))
    assert not p.controller.obstacle_stop_latched
    control=acc_tick(p,10.0)
    assert control.throttle>0 and control.brake==0
    p.controller.lidar_callback(lidar_cloud(3.8))
    control=acc_tick(p,10.0)
    assert control.throttle==0 and control.brake==1
