"""Stop-sign approach, full-stop dwell, and same-sign rearm suppression."""
import math


class StopSignPolicy:
    def __init__(self, stop_buffer=3.0, hold_seconds=3.0):
        self.stop_buffer = stop_buffer
        self.hold_seconds = hold_seconds
        self.distance = None
        self.last_seen = None
        self.last_tick = None
        self.stopped_since = None
        self.served = False
        self.travel_since_stop = 0.0
        self.confirmations = 0

    def observe(self, distance, now):
        if distance is None or not math.isfinite(distance) or distance <= 0:
            return
        if self.served:
            self.last_seen = now
            return
        self.confirmations = self.confirmations + 1 if self.last_seen is not None and 0 <= now - self.last_seen < 1.0 else 1
        self.last_seen = now
        if self.confirmations >= 2:
            self.distance = distance if self.distance is None else min(self.distance, distance)

    def limit(self, speed, now):
        dt = 0.0 if self.last_tick is None else max(0.0, min(0.5, now - self.last_tick))
        if self.last_tick is not None and now < self.last_tick:
            # A clock reset must never count as completed dwell/clearance.
            self.stopped_since = None
            self.last_seen = now if self.last_seen is not None else None
        self.last_tick = now
        if self.served:
            self.travel_since_stop += max(0.0, speed) * dt
            if self.travel_since_stop >= 10.0 and self.last_seen is not None and now - self.last_seen >= 5.0:
                self.served = False
                self.confirmations = 0
                self.distance = None
            return None, False
        if self.distance is None:
            return None, False
        self.distance = max(0.0, self.distance - max(0.0, speed) * dt)
        remaining = max(0.0, self.distance - self.stop_buffer)
        if remaining <= 0.2:
            if abs(speed) < 0.1:
                if self.stopped_since is None:
                    self.stopped_since = now
                if now - self.stopped_since >= self.hold_seconds:
                    self.served = True
                    self.travel_since_stop = 0.0
                    self.distance = None
                    self.stopped_since = None
                    return None, False
            else:
                self.stopped_since = None
            return 0.0, True
        return math.sqrt(2.0 * 1.5 * remaining), False
