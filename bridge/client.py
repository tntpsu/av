"""
Python client helper for Unity bridge communication.
Provides convenient methods for AV stack components to interact with Unity.
"""

import base64
import io
import os
import queue as _queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image


class UnityBridgeClient:
    """Client for communicating with Unity bridge server."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        use_raw_camera: Optional[bool] = None,
        *,
        async_trajectory_transport: bool = True,
        trust_proxy_env: bool = False,
    ):
        """
        Initialize Unity bridge client.

        Args:
            base_url: Base URL of the bridge server
            use_raw_camera: If True, use GET /api/camera/.../raw (no JPEG round-trip).
                If None, default from env AV_STACK_RAW_CAMERA (1/true = raw, 0/false = JPEG only).
            async_trajectory_transport: If True, POST /api/trajectory from a background thread
                (latest-wins) so the main loop is not blocked on visualization.
            trust_proxy_env: If False (default), set Session.trust_env=False on all bridge
                sessions so requests does not resolve OS proxies (macOS System Configuration /
                _scproxy) on every call to localhost. Set True only if the bridge is reached
                via HTTP(S)_PROXY.
        """
        self.base_url = base_url.rstrip("/")
        self._trust_proxy_env = bool(trust_proxy_env)
        self.session = self._new_bridge_session()
        # Second session for parallel vehicle GET (Session is not thread-safe across threads).
        self._vehicle_parallel_session = self._new_bridge_session()
        if use_raw_camera is None:
            v = os.environ.get("AV_STACK_RAW_CAMERA", "1").strip().lower()
            use_raw_camera = v not in ("0", "false", "no", "off")
        self.use_raw_camera = bool(use_raw_camera)

        # Fire-and-forget control sender.
        # set_control_command() enqueues and returns immediately (latest-wins,
        # maxsize=1).  A daemon thread drains the queue and does the actual
        # HTTP POST so the orchestrator is never blocked waiting for Unity.
        self._control_queue: _queue.Queue = _queue.Queue(maxsize=1)
        self._control_dropped: int = 0   # commands superseded before being sent
        self._control_errors: int = 0    # HTTP errors from the sender thread
        self._control_thread = threading.Thread(
            target=self._control_sender_loop,
            daemon=True,
            name="control-sender",
        )
        self._control_thread.start()

        self._async_trajectory = bool(async_trajectory_transport)
        self._trajectory_queue: Optional[_queue.Queue] = None
        self._trajectory_dropped: int = 0
        self._trajectory_errors: int = 0
        self._trajectory_thread: Optional[threading.Thread] = None
        if self._async_trajectory:
            self._trajectory_queue = _queue.Queue(maxsize=1)
            self._trajectory_thread = threading.Thread(
                target=self._trajectory_sender_loop,
                daemon=True,
                name="trajectory-sender",
            )
            self._trajectory_thread.start()

        # Optional background fetch of latest front camera (dedicated Session — not thread-safe with self.session).
        self._prefetch_camera_id = "front_center"
        self._cam_lock = threading.Lock()
        self._cam_slot: Optional[Tuple[np.ndarray, float, Optional[int], Dict]] = None
        self._cam_stop = threading.Event()
        self._cam_thread: Optional[threading.Thread] = None

    def _new_bridge_session(self) -> requests.Session:
        """Session for bridge HTTP; trust_env off by default to avoid per-request proxy lookup."""
        sess = requests.Session()
        if not self._trust_proxy_env:
            sess.trust_env = False
        return sess

    @staticmethod
    def _parse_raw_camera_response(response: requests.Response) -> Tuple[np.ndarray, float, Optional[int], Dict]:
        """Decode octet-stream body + X-AV-* headers from /api/camera/*/raw."""
        shape_s = (response.headers.get("X-AV-Shape") or "").strip()
        if not shape_s:
            raise ValueError("raw camera response missing X-AV-Shape")
        parts = [int(x.strip()) for x in shape_s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"invalid X-AV-Shape: {shape_s!r}")
        h, w, c = parts
        buf = response.content
        expected = h * w * c
        if len(buf) != expected:
            raise ValueError(f"raw camera size mismatch: got {len(buf)} expected {expected}")
        arr = np.frombuffer(memoryview(buf), dtype=np.uint8).reshape((h, w, c)).copy()

        ts_s = (response.headers.get("X-AV-Timestamp") or "").strip()
        timestamp = float(ts_s) if ts_s else time.time()

        fid_s = (response.headers.get("X-AV-Frame-Id") or "").strip()
        frame_id: Optional[int] = int(fid_s) if fid_s else None

        def _int_hdr(name: str) -> Optional[int]:
            s = (response.headers.get(name) or "").strip()
            if not s:
                return None
            try:
                return int(s)
            except ValueError:
                return None

        def _float_hdr(name: str) -> Optional[float]:
            s = (response.headers.get(name) or "").strip()
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None

        meta: Dict = {
            "queue_depth": _int_hdr("X-AV-Queue-Depth"),
            "queue_capacity": _int_hdr("X-AV-Queue-Capacity"),
            "drop_count": _int_hdr("X-AV-Drop-Count"),
            "decode_in_flight": (response.headers.get("X-AV-Decode-In-Flight") or "").strip() == "1",
            "latest_age_ms": _float_hdr("X-AV-Latest-Age-Ms"),
            "last_arrival_time": _float_hdr("X-AV-Last-Arrival-Time"),
            "last_realtime_since_startup": _float_hdr("X-AV-Last-Realtime"),
            "last_unscaled_time": _float_hdr("X-AV-Last-Unscaled"),
            "queue_remaining": _int_hdr("X-AV-Queue-Remaining"),
        }
        return arr, timestamp, frame_id, meta

    def start_camera_prefetch(self, camera_id: str = "front_center") -> None:
        """Start a daemon thread that continuously GETs /api/camera/latest(/raw) for ``camera_id``.

        The main loop then takes the freshest completed fetch from a slot (latest-wins), overlapping
        HTTP+decode with perception/control work. Uses a **separate** ``requests.Session`` in the worker.
        """
        self._prefetch_camera_id = camera_id
        if self._cam_thread is not None and self._cam_thread.is_alive():
            return
        self._cam_stop.clear()
        self._cam_thread = threading.Thread(
            target=self._camera_prefetch_loop,
            daemon=True,
            name="camera-prefetch",
        )
        self._cam_thread.start()

    def stop_camera_prefetch(self) -> None:
        """Stop the prefetch worker and clear the slot."""
        self._cam_stop.set()
        t = self._cam_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._cam_thread = None
        with self._cam_lock:
            self._cam_slot = None

    def _camera_prefetch_loop(self) -> None:
        worker_session = self._new_bridge_session()
        cam = self._prefetch_camera_id
        while not self._cam_stop.is_set():
            try:
                data = self._sync_get_latest_camera_frame_with_metadata(cam, session=worker_session)
                if data is not None:
                    with self._cam_lock:
                        self._cam_slot = data
                else:
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.02)

    def drain_camera_queue(self, camera_id: str = "front_center") -> Optional[int]:
        """POST /api/camera/drain_queue — clear FIFO backlog; ``latest`` snapshot unchanged."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/camera/drain_queue",
                params={"camera_id": camera_id},
                timeout=2.0,
            )
            if response.status_code == 200:
                body = response.json()
                return int(body.get("drained", 0))
        except requests.RequestException:
            pass
        return None

    def _sync_get_latest_camera_frame_with_metadata(
        self,
        camera_id: str,
        *,
        session: requests.Session,
    ) -> Optional[Tuple[np.ndarray, float, Optional[int], Dict]]:
        """Synchronous HTTP fetch (used by main thread and prefetch worker)."""
        try:
            if self.use_raw_camera:
                raw_resp = session.get(
                    f"{self.base_url}/api/camera/latest/raw",
                    params={"camera_id": camera_id},
                    timeout=0.5,
                )
                if raw_resp.status_code == 200:
                    try:
                        return self._parse_raw_camera_response(raw_resp)
                    except ValueError:
                        pass
                elif raw_resp.status_code != 404:
                    raw_resp.raise_for_status()

            response = session.get(
                f"{self.base_url}/api/camera/latest",
                params={"camera_id": camera_id},
                timeout=0.5,
            )
            response.raise_for_status()

            data = response.json()
            img_base64 = data["image"]
            timestamp = data.get("timestamp", time.time())
            frame_id = data.get("frame_id")

            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)

            meta = {
                "queue_depth": data.get("queue_depth"),
                "queue_capacity": data.get("queue_capacity"),
                "drop_count": data.get("drop_count"),
                "decode_in_flight": data.get("decode_in_flight"),
                "latest_age_ms": data.get("latest_age_ms"),
                "last_arrival_time": data.get("last_arrival_time"),
                "last_realtime_since_startup": data.get("last_realtime_since_startup"),
                "last_unscaled_time": data.get("last_unscaled_time"),
            }
            return img_array, timestamp, frame_id, meta
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None

    def get_latest_camera_frame(
        self,
        camera_id: str = "front_center"
    ) -> Optional[Tuple[np.ndarray, float, Optional[int]]]:
        """
        Get latest camera frame from Unity.
        
        Returns:
            Tuple of (image_array, timestamp) or None if not available
        """
        frame = self.get_latest_camera_frame_with_metadata(camera_id=camera_id)
        if frame is None:
            return None
        image, timestamp, frame_id, _meta = frame
        return image, timestamp, frame_id

    def get_latest_camera_frame_with_metadata(
        self,
        camera_id: str = "front_center"
    ) -> Optional[Tuple[np.ndarray, float, Optional[int], Dict]]:
        """Get latest camera frame plus bridge freshness metadata.

        When :meth:`start_camera_prefetch` is running for this ``camera_id``, returns the freshest
        prefetched frame (non-blocking aside from a short lock); otherwise performs a sync GET.
        """
        if (
            self._cam_thread is not None
            and self._cam_thread.is_alive()
            and camera_id == self._prefetch_camera_id
        ):
            with self._cam_lock:
                slot = self._cam_slot
                self._cam_slot = None
            if slot is not None:
                return slot

        return self._sync_get_latest_camera_frame_with_metadata(camera_id, session=self.session)

    def get_next_camera_frame(
        self,
        camera_id: str = "front_center"
    ) -> Optional[Tuple[np.ndarray, float, Optional[int]]]:
        """
        Get next queued camera frame from Unity (FIFO order).

        Returns:
            Tuple of (image_array, timestamp, frame_id) or None if unavailable
        """
        try:
            if self.use_raw_camera:
                raw_resp = self.session.get(
                    f"{self.base_url}/api/camera/next/raw",
                    params={"camera_id": camera_id},
                    timeout=0.5,
                )
                if raw_resp.status_code == 200:
                    try:
                        arr, ts, fid, _meta = self._parse_raw_camera_response(raw_resp)
                        return arr, ts, fid
                    except ValueError:
                        pass
                elif raw_resp.status_code != 404:
                    raw_resp.raise_for_status()

            response = self.session.get(
                f"{self.base_url}/api/camera/next",
                params={"camera_id": camera_id},
                timeout=0.5,
            )
            response.raise_for_status()

            data = response.json()
            img_base64 = data["image"]
            timestamp = data.get("timestamp", time.time())
            frame_id = data.get("frame_id")

            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)

            return img_array, timestamp, frame_id
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None
    
    def _sync_get_latest_vehicle_state(self, session: requests.Session) -> Optional[Dict]:
        """GET /api/vehicle/state/latest using the given Session (thread-local for parallel fetch)."""
        try:
            response = session.get(
                f"{self.base_url}/api/vehicle/state/latest",
                timeout=0.5,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None

    def get_latest_vehicle_state(self) -> Optional[Dict]:
        """
        Get latest vehicle state from Unity.

        Returns:
            Vehicle state dictionary or None if not available
        """
        return self._sync_get_latest_vehicle_state(self.session)

    def get_latest_camera_and_vehicle_parallel(
        self,
        camera_id: str = "front_center",
    ) -> Tuple[
        Optional[Tuple[np.ndarray, float, Optional[int], Dict]],
        Optional[Dict],
        float,
        float,
    ]:
        """Overlap vehicle state GET with camera fetch (separate Session for vehicle).

        Submits vehicle fetch on a worker thread, then runs
        :meth:`get_latest_camera_frame_with_metadata` on the main thread (uses ``self.session``).
        Returns monotonic completion times for camera and vehicle; vehicle time is recorded
        in the worker when the HTTP response is received.

        Returns:
            (frame_data, vehicle_state_dict, front_ready_mono_s, vehicle_ready_mono_s)
        """
        from concurrent.futures import ThreadPoolExecutor

        def _vehicle_worker() -> Tuple[Optional[Dict], float]:
            d = self._sync_get_latest_vehicle_state(self._vehicle_parallel_session)
            return d, time.monotonic()

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_vehicle_worker)
            frame_data = self.get_latest_camera_frame_with_metadata(camera_id=camera_id)
            front_ready_mono_s = time.monotonic()
            vehicle_state_dict, vehicle_ready_mono_s = fut.result()
        return frame_data, vehicle_state_dict, front_ready_mono_s, vehicle_ready_mono_s

    def get_next_vehicle_state(self) -> Optional[Dict]:
        """
        Get next queued vehicle state from Unity bridge (FIFO order).

        Returns:
            Vehicle state dictionary or None if unavailable
        """
        try:
            response = self.session.get(f"{self.base_url}/api/vehicle/state/next", timeout=0.5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None
    
    def get_latest_unity_feedback(self) -> Optional[Dict]:
        """
        Get latest Unity feedback/status data.
        
        Returns:
            Unity feedback dictionary or None if not available
        """
        try:
            response = self.session.get(f"{self.base_url}/api/unity/feedback/latest", timeout=0.5)
            response.raise_for_status()
            data = response.json()
            return data if data else None
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None
    
    def _build_control_command(
        self,
        steering: float,
        throttle: float,
        brake: float,
        ground_truth_mode: bool = False,
        ground_truth_speed: float = 5.0,
        randomize_start: bool = False,
        randomize_request_id: int = 0,
        randomize_seed: int | None = None,
    ) -> dict:
        command = {
            "steering": float(steering),
            "throttle": float(throttle),
            "brake": float(brake),
            "ground_truth_mode": bool(ground_truth_mode),
            "ground_truth_speed": float(ground_truth_speed),
            "randomize_start": bool(randomize_start),
            "randomize_request_id": int(randomize_request_id),
        }
        if randomize_seed is not None:
            command["randomize_seed"] = int(randomize_seed)
        return command

    # ── Background control sender ─────────────────────────────────────────────

    def _control_sender_loop(self) -> None:
        """Daemon thread: POSTs control commands from _control_queue to Unity.

        Runs until a None sentinel is enqueued (via close()).
        """
        while True:
            try:
                command = self._control_queue.get(timeout=1.0)
            except _queue.Empty:
                continue
            if command is None:   # shutdown sentinel
                break
            try:
                self.session.post(
                    f"{self.base_url}/api/vehicle/control",
                    json=command,
                    timeout=0.5,
                )
            except requests.RequestException:
                self._control_errors += 1

    def _trajectory_sender_loop(self) -> None:
        """Daemon thread: POSTs trajectory JSON from queue to Unity (latest-wins)."""
        assert self._trajectory_queue is not None
        while True:
            try:
                item = self._trajectory_queue.get(timeout=1.0)
            except _queue.Empty:
                continue
            if item is None:
                break
            try:
                self._sync_set_trajectory_data(item)
            except requests.RequestException:
                self._trajectory_errors += 1

    def _sync_set_trajectory_data(self, trajectory_data: Dict[str, Any]) -> bool:
        """Synchronous POST /api/trajectory (used by main thread or trajectory sender)."""
        response = self.session.post(
            f"{self.base_url}/api/trajectory",
            json=trajectory_data,
            timeout=2.0,
        )
        response.raise_for_status()
        return True

    def close(self) -> None:
        """Shut down background workers and close HTTP sessions."""
        self.stop_camera_prefetch()
        try:
            self._control_queue.put_nowait(None)
        except _queue.Full:
            pass
        if self._trajectory_queue is not None:
            try:
                try:
                    self._trajectory_queue.get_nowait()
                except _queue.Empty:
                    pass
                self._trajectory_queue.put_nowait(None)
            except _queue.Full:
                pass
        try:
            self._control_thread.join(timeout=2.0)
        except Exception:
            pass
        if self._trajectory_thread is not None:
            try:
                self._trajectory_thread.join(timeout=2.0)
            except Exception:
                pass
        try:
            self.session.close()
        except Exception:
            pass
        try:
            self._vehicle_parallel_session.close()
        except Exception:
            pass

    def set_control_command(
        self,
        steering: float,
        throttle: float,
        brake: float,
        ground_truth_mode: bool = False,
        ground_truth_speed: float = 5.0,
        randomize_start: bool = False,
        randomize_request_id: int = 0,
        randomize_seed: int | None = None,
    ) -> bool:
        """Enqueue a control command for fire-and-forget delivery to Unity.

        Returns immediately without blocking on the HTTP round-trip.
        The background sender thread delivers the command; latest-wins if
        a previous command has not yet been sent.

        Args:
            steering: Steering angle (-1.0 to 1.0)
            throttle: Throttle (-1.0 to 1.0, negative = reverse)
            brake: Brake (0.0 to 1.0)
            ground_truth_mode: Enable direct velocity control
            ground_truth_speed: Speed for ground truth mode (m/s)

        Returns:
            True always (enqueue cannot fail; HTTP errors are logged silently).
        """
        command = self._build_control_command(
            steering=steering,
            throttle=throttle,
            brake=brake,
            ground_truth_mode=ground_truth_mode,
            ground_truth_speed=ground_truth_speed,
            randomize_start=randomize_start,
            randomize_request_id=randomize_request_id,
            randomize_seed=randomize_seed,
        )
        # Latest-wins: drain any unsent command, then enqueue the new one.
        try:
            self._control_queue.get_nowait()
            self._control_dropped += 1
        except _queue.Empty:
            pass
        self._control_queue.put_nowait(command)
        return True
    
    def request_unity_play(self) -> bool:
        """
        Request Unity to start playing (enter play mode).
        Only works when Unity Editor is open.
        
        Returns:
            True if request was sent successfully, False otherwise
        """
        try:
            response = self.session.post(f"{self.base_url}/api/unity/play", timeout=2.0)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "play_requested":
                return True
            return False
        except requests.RequestException as e:
            print(f"Error requesting Unity play: {e}")
            return False
    
    def set_trajectory_data(
        self,
        trajectory_points: list,
        reference_point: list,
        lateral_error: float,
        perception_left_lane_x: float | None = None,
        perception_right_lane_x: float | None = None,
        perception_center_x: float | None = None,
        perception_lookahead_m: float | None = None,
        perception_valid: bool | None = None,
    ) -> bool:
        """
        Set trajectory data for Unity visualization.

        When async_trajectory_transport is True (default), enqueues for the background
        sender (latest-wins) and returns immediately.

        Args:
            trajectory_points: List of [x, y, heading] points
            reference_point: [x, y, heading, velocity] reference point
            lateral_error: Current lateral error for color coding

        Returns:
            True if enqueued or sent successfully, False otherwise
        """
        trajectory_data: Dict[str, Any] = {
            "trajectory_points": trajectory_points,
            "reference_point": reference_point,
            "lateral_error": float(lateral_error),
            "timestamp": time.time(),
        }
        if perception_left_lane_x is not None:
            trajectory_data["perception_left_lane_x"] = float(perception_left_lane_x)
        if perception_right_lane_x is not None:
            trajectory_data["perception_right_lane_x"] = float(perception_right_lane_x)
        if perception_center_x is not None:
            trajectory_data["perception_center_x"] = float(perception_center_x)
        if perception_lookahead_m is not None:
            trajectory_data["perception_lookahead_m"] = float(perception_lookahead_m)
        if perception_valid is not None:
            trajectory_data["perception_valid"] = bool(perception_valid)

        if self._async_trajectory:
            if self._trajectory_queue is None:
                return False
            try:
                try:
                    self._trajectory_queue.get_nowait()
                    self._trajectory_dropped += 1
                except _queue.Empty:
                    pass
                self._trajectory_queue.put_nowait(trajectory_data)
                return True
            except _queue.Full:
                self._trajectory_dropped += 1
                return False

        try:
            return self._sync_set_trajectory_data(trajectory_data)
        except requests.RequestException:
            # Don't log errors - trajectory visualization is optional
            return False
    
    def health_check(self) -> bool:
        """
        Check if bridge server is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=1.0)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False
    
    def signal_shutdown(self) -> bool:
        """
        Signal to Unity that the AV stack is shutting down.
        Unity will exit play mode when it detects this signal.
        
        Returns:
            True if signal was sent successfully, False otherwise
        """
        try:
            response = self.session.post(f"{self.base_url}/api/shutdown", timeout=2.0)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "shutdown":
                return True
            return False
        except requests.RequestException as e:
            print(f"Error signaling shutdown: {e}")
            return False


if __name__ == "__main__":
    # Test client
    client = UnityBridgeClient()
    
    if client.health_check():
        print("Bridge server is healthy")
        
        # Get camera frame
        frame_data = client.get_latest_camera_frame()
        if frame_data:
            img, timestamp = frame_data
            print(f"Got camera frame: {img.shape}, timestamp: {timestamp}")
        
        # Get vehicle state
        state = client.get_latest_vehicle_state()
        if state:
            print(f"Got vehicle state: speed={state.get('speed', 0):.2f} m/s")
        
        # Set control command
        client.set_control_command(steering=0.1, throttle=0.5, brake=0.0)
        print("Set control command")
    else:
        print("Bridge server is not available")

