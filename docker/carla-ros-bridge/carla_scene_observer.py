#!/usr/bin/env python3
"""Read-only CARLA -> ROS scene facts. Run in the CARLA bridge's Python runtime.
Never ticks, spawns, destroys, teleports, or controls actors.
"""
import argparse
import json
import math
import os
import time


def distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)


def scene(world, ego):
    transform = ego.get_transform()
    loc = transform.location
    forward = transform.get_forward_vector()
    front = ego.bounding_box.extent.x
    width = ego.bounding_box.extent.y
    waypoint = world.get_map().get_waypoint(loc)
    route = [(0.0, waypoint)] if waypoint else []
    # Trace the current lane, retaining branches at intersections. Match stop
    # waypoints/trigger volumes against lane geometry, not the sign's appearance.
    frontier = [waypoint] if waypoint else []
    seen = set()
    for step in range(1, 61):
        next_frontier = []
        for wp in frontier:
            for nxt in wp.next(0.5):
                key = (nxt.road_id, nxt.lane_id, round(nxt.s, 1))
                if key in seen:
                    continue
                seen.add(key)
                route.append((step*0.5, nxt))
                next_frontier.append(nxt)
        frontier = next_frontier[:12]
        if not frontier:
            break
    actors = world.get_actors()
    stop_distance = None
    stop_id = None
    for actor in actors.filter('traffic.stop*'):
        tf = actor.get_transform()
        trigger = actor.trigger_volume
        inverse = tf.get_inverse_matrix()
        candidates = []
        for d, wp in route:
            point = wp.transform.location
            v = [point.x, point.y, point.z, 1.0]
            local = [sum(inverse[i][j]*v[j] for j in range(4)) for i in range(3)]
            x, y = local[0]-trigger.location.x, local[1]-trigger.location.y
            angle = math.radians(trigger.rotation.yaw)
            bx, by = x*math.cos(angle)+y*math.sin(angle), -x*math.sin(angle)+y*math.cos(angle)
            if abs(bx) <= trigger.extent.x+0.25 and abs(by) <= trigger.extent.y+0.25:
                candidates.append(d)
        if candidates:
            d = min(candidates)
            if stop_distance is None or d < stop_distance:
                stop_distance, stop_id = d, actor.id
    light = ego.get_traffic_light()
    candidates = []
    if waypoint and hasattr(world, 'get_traffic_lights_from_waypoint'):
        candidates = list(world.get_traffic_lights_from_waypoint(waypoint, 30.0))
    if light and all(a.id != light.id for a in candidates):
        candidates.append(light)
    lights = []
    for actor in candidates:
        distances = []
        for stop in actor.get_stop_waypoints():
            for d, wp in route:
                if wp.road_id == stop.road_id and wp.lane_id == stop.lane_id and distance(wp.transform.location, stop.transform.location) < 1.0:
                    distances.append(d)
        if distances:
            lights.append((min(distances), str(actor.get_state()).split('.')[-1], actor.id))
        elif light and actor.id == light.id:
            lights.append((front+2.0, str(actor.get_state()).split('.')[-1], actor.id))
    light_info = min(lights) if lights else (None, 'None', None)
    gap = None
    for actor in actors.filter('vehicle.*'):
        if actor.id == ego.id:
            continue
        other_tf = actor.get_transform()
        other = other_tf.transform(actor.bounding_box.location)
        dx, dy = other.x-loc.x, other.y-loc.y
        x = dx*forward.x + dy*forward.y
        y = -dx*forward.y + dy*forward.x
        angle = math.radians(other_tf.rotation.yaw-transform.rotation.yaw)
        ex, ey = actor.bounding_box.extent.x, actor.bounding_box.extent.y
        longitudinal = abs(math.cos(angle))*ex + abs(math.sin(angle))*ey
        lateral = abs(math.sin(angle))*ex + abs(math.cos(angle))*ey
        if x + longitudinal <= 0 or abs(other.z-loc.z) > 3.0:
            continue
        on_lane = abs(y) <= width+lateral+0.3
        # Also protect a curved lane through a turn.
        if not on_lane:
            on_lane = any(d < 15 and distance(wp.transform.location, other) < ex+width+0.3 for d, wp in route)
        if on_lane:
            candidate = max(0.0, x-longitudinal-front) if abs(y) <= width+lateral+0.3 else max(0.0, distance(loc,other)-ex-front)
            gap = candidate if gap is None else min(gap, candidate)
    return {'valid': True, 'stamp': world.get_snapshot().timestamp.elapsed_seconds,
            'ego_id': ego.id, 'ego_front': front, 'vehicle_gap': gap,
            'stop_distance': stop_distance, 'stop_id': stop_id,
            'light_distance': light_info[0], 'light_state': light_info[1], 'light_id': light_info[2]}


class SceneReader:
    """Discard failed CARLA sessions so a restarted server can reconnect."""

    def __init__(self, client_factory, host, port):
        self.client_factory = client_factory
        self.host = host
        self.port = port
        self.client = None

    def read(self):
        try:
            if self.client is None:
                self.client = self.client_factory(self.host, self.port)
                self.client.set_timeout(1.0)
            world = self.client.get_world()
            ego = next((a for a in world.get_actors().filter('vehicle.*')
                        if a.attributes.get('role_name') == 'hero'), None)
            return scene(world, ego) if ego else {'valid': False, 'error': 'No hero actor'}
        except Exception as error:
            # A client whose simulator session has died can keep timing out
            # after the server returns. Recreate it on the next polling cycle.
            self.client = None
            return {'valid': False, 'error': str(error)}


def main():
    import carla
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=2000+int(os.environ.get('ROS_DOMAIN_ID', '83')))
    parser.add_argument('--host', default='localhost')
    args = parser.parse_args()
    rclpy.init(args=[])
    node = Node('carla_scene_observer')
    pub = node.create_publisher(String, '/carla/scene', 10)
    reader = SceneReader(carla.Client, args.host, args.port)
    try:
        while rclpy.ok():
            data = reader.read()
            pub.publish(String(data=json.dumps(data)))
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
