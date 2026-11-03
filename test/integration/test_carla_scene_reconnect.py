"""The scene observer must recover when CARLA restarts beneath it."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def observer_module():
    path = Path(__file__).resolve().parents[2] / 'docker/carla-ros-bridge/carla_scene_observer.py'
    spec = importlib.util.spec_from_file_location('scene_reconnect_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_session_is_replaced_after_server_restart(monkeypatch):
    module = observer_module()
    ego = SimpleNamespace(attributes={'role_name': 'hero'})
    world = SimpleNamespace(get_actors=lambda: SimpleNamespace(filter=lambda pattern: [ego]))
    created = []

    class Client:
        def __init__(self, host, port):
            assert (host, port) == ('localhost', 2083)
            self.failed_session = not created
            created.append(self)

        def set_timeout(self, value):
            assert value == 1.0

        def get_world(self):
            if self.failed_session:
                raise RuntimeError('simulator connection timed out')
            return world

    monkeypatch.setattr(module, 'scene', lambda w, e: {'valid': True, 'light_state': 'Green'})
    reader = module.SceneReader(Client, 'localhost', 2083)
    assert reader.read()['valid'] is False
    assert reader.client is None
    assert reader.read() == {'valid': True, 'light_state': 'Green'}
    assert reader.read()['valid'] is True
    assert len(created) == 2


def test_missing_ego_recovers_without_churning_healthy_connection(monkeypatch):
    module = observer_module()
    actors = []
    created = []
    world = SimpleNamespace(get_actors=lambda: SimpleNamespace(filter=lambda pattern: actors))

    def client_factory(host, port):
        client = SimpleNamespace(set_timeout=lambda value: None, get_world=lambda: world)
        created.append(client)
        return client

    monkeypatch.setattr(module, 'scene', lambda w, e: {'valid': True})
    reader = module.SceneReader(client_factory, 'localhost', 2083)
    assert reader.read() == {'valid': False, 'error': 'No hero actor'}
    actors.append(SimpleNamespace(attributes={'role_name': 'hero'}))
    assert reader.read()['valid'] is True
    assert len(created) == 1
