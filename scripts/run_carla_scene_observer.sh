#!/usr/bin/env bash
set -euo pipefail
bridge_container=${1:?Usage: run_carla_scene_observer.sh BRIDGE_CONTAINER}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
docker cp "$repo_dir/docker/carla-ros-bridge/carla_scene_observer.py" "$bridge_container:/tmp/carla_scene_observer.py"
docker exec -d -e ROS_DOMAIN_ID=83 "$bridge_container" bash -lc '
if pgrep -f "^python3 /tmp/carla_scene_observer.py" >/dev/null; then exit 0; fi
source /opt/ros/foxy/setup.bash
export PYTHONPATH=/carla/PythonAPI/carla:/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:$PYTHONPATH
exec python3 /tmp/carla_scene_observer.py --port 2083 >/tmp/carla_scene_observer.log 2>&1
'
