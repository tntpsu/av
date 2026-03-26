"""
Python client helper for Unity bridge communication.
Provides convenient methods for AV stack components to interact with Unity.
"""

import base64
import io
import json
import os
import queue as _queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

import concurrent.futures

import numpy as np
import requests
from PIL import Image

from sync_contract import (
    PACKET_CONSUME_POLICY_FIFO_STRICT,
    PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
    PACKET_CONSUME_POLICY_LATEST_DEBUG,
    PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED,
    PACKET_FALLBACK_PAYLOAD_DRAIN_CAP,
    PACKET_FALLBACK_PAYLOAD_STALE_TIMEOUT,
    PACKET_FALLBACK_MISSING_FRONT_CAMERA,
    PACKET_FALLBACK_MISSING_VEHICLE_STATE,
    PACKET_FALLBACK_NONE,
    PACKET_FALLBACK_PACKET_LEGACY_PATH,
    PACKET_FALLBACK_PACKET_QUEUE_OVERFLOW,
    PACKET_FALLBACK_PACKET_TIMEOUT,
    PACKET_MODE_LATEST_PARALLEL,
)


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
        # Persistent executor for parallel vehicle fetch — reused every frame to avoid
        # per-call thread creation/join overhead (~20-80 ms on macOS per call).
        self._parallel_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vehicle-parallel"
        )
        if use_raw_camera is None:
            v = os.environ.get("AV_STACK_RAW_CAMERA", "1").strip().lower()
            use_raw_camera = v not in ("0", "false", "no", "off")
        self.use_raw_camera = bool(use_raw_camera)
        self.sync_packet_mode = PACKET_MODE_LATEST_PARALLEL

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
            "unity_frame_count": _int_hdr("X-AV-Unity-Frame-Count"),
            "unity_time": _float_hdr("X-AV-Unity-Time"),
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
                "unity_frame_count": data.get("unity_frame_count"),
                "unity_time": data.get("unity_time"),
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

    def get_next_camera_frame_with_metadata(
        self,
        camera_id: str = "front_center",
    ) -> Optional[Tuple[np.ndarray, float, Optional[int], Dict]]:
        """Get next queued camera frame plus bridge freshness metadata."""
        try:
            if self.use_raw_camera:
                raw_resp = self.session.get(
                    f"{self.base_url}/api/camera/next/raw",
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

            meta = {
                "unity_frame_count": data.get("unity_frame_count"),
                "unity_time": data.get("unity_time"),
                "queue_remaining": data.get("queue_remaining"),
            }
            return img_array, timestamp, frame_id, meta
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

    def _sync_get_latest_sync_packet(self, session: requests.Session) -> Optional[Dict]:
        """GET /api/sim/packet/latest using the given Session."""
        try:
            response = session.get(
                f"{self.base_url}/api/sim/packet/latest",
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

    def get_latest_sync_packet(self) -> Optional[Dict]:
        """Get latest synchronized bridge packet metadata."""
        return self._sync_get_latest_sync_packet(self.session)

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
        def _vehicle_worker() -> Tuple[Optional[Dict], float]:
            d = self._sync_get_latest_vehicle_state(self._vehicle_parallel_session)
            return d, time.monotonic()

        fut = self._parallel_executor.submit(_vehicle_worker)
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

    def get_next_sync_packet(self) -> Optional[Dict]:
        """Get next synchronized bridge packet metadata (FIFO order)."""
        try:
            response = self.session.get(f"{self.base_url}/api/sim/packet/next", timeout=0.5)
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

    @staticmethod
    def _extract_unity_frame_count_from_vehicle_state(vehicle_state: Optional[Dict]) -> Optional[int]:
        if not isinstance(vehicle_state, dict):
            return None
        raw = vehicle_state.get("unityFrameCount", vehicle_state.get("unity_frame_count"))
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_unity_frame_count_from_camera_frame(
        frame_data: Optional[Tuple[np.ndarray, float, Optional[int], Dict]],
    ) -> Optional[int]:
        if frame_data is None or len(frame_data) < 4:
            return None
        meta = frame_data[3] or {}
        raw = meta.get("unity_frame_count")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def get_next_sync_packet_bundle(
        self,
        camera_id: str = "front_center",
        *,
        max_resync_discards: int = 8,
        drain_to_latest: bool = False,
        max_packet_drains: int = 256,
    ) -> Optional[Dict[str, Any]]:
        """Consume the next synchronized packet and align FIFO camera/state components to it."""
        packet = self.get_next_sync_packet()
        if packet is None:
            return None
        drained_packet_count = 0
        if drain_to_latest:
            for _ in range(max(0, int(max_packet_drains))):
                newer = self.get_next_sync_packet()
                if newer is None:
                    break
                packet = newer
                drained_packet_count += 1

        expected_unity_frame = packet.get("unity_frame_count")
        try:
            expected_unity_frame = int(expected_unity_frame) if expected_unity_frame is not None else None
        except (TypeError, ValueError):
            expected_unity_frame = None

        bundle: Dict[str, Any] = {
            "packet": packet,
            "frame_data": None,
            "vehicle_state": None,
            "front_ready_mono_s": None,
            "vehicle_ready_mono_s": None,
            "discarded_camera_frames": 0,
            "discarded_vehicle_states": 0,
            "drained_packet_count": drained_packet_count,
            "fallback_active": False,
            "fallback_reason_code": PACKET_FALLBACK_NONE,
        }

        camera_frame = None
        for _ in range(max(1, int(max_resync_discards) + 1)):
            camera_frame = self.get_next_camera_frame_with_metadata(camera_id=camera_id)
            bundle["front_ready_mono_s"] = time.monotonic()
            if camera_frame is None:
                bundle["fallback_active"] = True
                bundle["fallback_reason_code"] = PACKET_FALLBACK_MISSING_FRONT_CAMERA
                break
            camera_unity_frame = self._extract_unity_frame_count_from_camera_frame(camera_frame)
            if expected_unity_frame is None or camera_unity_frame is None or camera_unity_frame == expected_unity_frame:
                break
            if camera_unity_frame < expected_unity_frame:
                bundle["discarded_camera_frames"] += 1
                camera_frame = None
                continue
            bundle["fallback_active"] = True
            bundle["fallback_reason_code"] = PACKET_FALLBACK_PACKET_QUEUE_OVERFLOW
            break
        bundle["frame_data"] = camera_frame

        vehicle_state = None
        for _ in range(max(1, int(max_resync_discards) + 1)):
            vehicle_state = self.get_next_vehicle_state()
            bundle["vehicle_ready_mono_s"] = time.monotonic()
            if vehicle_state is None:
                if not bundle["fallback_active"]:
                    bundle["fallback_active"] = True
                    bundle["fallback_reason_code"] = PACKET_FALLBACK_MISSING_VEHICLE_STATE
                break
            vehicle_unity_frame = self._extract_unity_frame_count_from_vehicle_state(vehicle_state)
            if expected_unity_frame is None or vehicle_unity_frame is None or vehicle_unity_frame == expected_unity_frame:
                break
            if vehicle_unity_frame < expected_unity_frame:
                bundle["discarded_vehicle_states"] += 1
                vehicle_state = None
                continue
            bundle["fallback_active"] = True
            bundle["fallback_reason_code"] = PACKET_FALLBACK_PACKET_QUEUE_OVERFLOW
            break
        bundle["vehicle_state"] = vehicle_state

        if bundle["frame_data"] is None and not bundle["fallback_active"]:
            bundle["fallback_active"] = True
            bundle["fallback_reason_code"] = PACKET_FALLBACK_PACKET_TIMEOUT
        if bundle["vehicle_state"] is None and not bundle["fallback_active"]:
            bundle["fallback_active"] = True
            bundle["fallback_reason_code"] = PACKET_FALLBACK_PACKET_TIMEOUT

        return bundle

    @staticmethod
    def _parse_multipart_parts(body: bytes, boundary: str) -> list[tuple[dict[str, str], bytes]]:
        delimiter = f"--{boundary}".encode("utf-8")
        parts: list[tuple[dict[str, str], bytes]] = []
        for raw_part in body.split(delimiter):
            part = raw_part.strip()
            if not part or part == b"--":
                continue
            if part.endswith(b"--"):
                part = part[:-2].rstrip()
            if b"\r\n\r\n" not in part:
                continue
            header_bytes, payload = part.split(b"\r\n\r\n", 1)
            headers: dict[str, str] = {}
            for line in header_bytes.split(b"\r\n"):
                if b":" not in line:
                    continue
                k, v = line.split(b":", 1)
                headers[k.decode("utf-8", errors="ignore").strip().lower()] = (
                    v.decode("utf-8", errors="ignore").strip()
                )
            parts.append((headers, payload.rstrip(b"\r\n")))
        return parts

    def _parse_sync_packet_payload_response(self, response: requests.Response) -> Dict[str, Any]:
        content_type = response.headers.get("Content-Type", "")
        boundary_token = "boundary="
        if boundary_token not in content_type:
            raise ValueError("sync packet payload missing multipart boundary")
        boundary = content_type.split(boundary_token, 1)[1].strip().strip('"')
        metadata: Optional[Dict[str, Any]] = None
        image_bytes: Optional[bytes] = None
        for headers, payload in self._parse_multipart_parts(response.content, boundary):
            part_type = headers.get("content-type", "")
            if part_type.startswith("application/json"):
                metadata = json.loads(payload.decode("utf-8"))
            elif part_type.startswith("application/octet-stream"):
                image_bytes = payload
        if not isinstance(metadata, dict):
            raise ValueError("sync packet payload missing metadata part")
        if image_bytes is None:
            raise ValueError("sync packet payload missing image bytes part")
        front = metadata.get("front_camera") or {}
        shape = front.get("shape")
        dtype_name = front.get("dtype")
        if not isinstance(shape, list) or not dtype_name:
            raise ValueError("sync packet payload missing image shape/dtype")
        dtype = np.dtype(str(dtype_name))
        expected = int(np.prod(shape)) * int(dtype.itemsize)
        if len(image_bytes) != expected:
            raise ValueError(
                f"sync packet payload image size mismatch: got {len(image_bytes)} expected {expected}"
            )
        image = np.frombuffer(memoryview(image_bytes), dtype=dtype).reshape(tuple(shape)).copy()
        packet_meta = metadata.get("packet") or {}
        vehicle_state = metadata.get("vehicle_state") or {}
        timestamp = front.get("timestamp", time.time())
        frame_id = front.get("frame_id")
        camera_meta = {
            "unity_frame_count": front.get("unity_frame_count"),
            "unity_time": front.get("unity_time"),
            "camera_id": front.get("camera_id"),
        }
        ready = time.monotonic()
        return {
            "image": image,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "camera_meta": camera_meta,
            "vehicle_state_dict": vehicle_state,
            "packet_meta": packet_meta,
            "fallback_active": bool(packet_meta.get("fallback_active", False)),
            "fallback_reason_code": str(packet_meta.get("fallback_reason_code") or ""),
            "front_ready_mono_s": ready,
            "vehicle_ready_mono_s": ready,
            "inputs_ready_mono_s": ready,
        }

    @staticmethod
    def _extract_sync_packet_age_ms(payload: Optional[Dict[str, Any]]) -> float:
        if not isinstance(payload, dict):
            return float("nan")
        packet_meta = payload.get("packet_meta") or {}
        try:
            value = float(packet_meta.get("packet_age_ms", float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    @staticmethod
    def _parse_selection_headers(
        headers: Dict[str, Any],
        *,
        consume_policy: str,
        max_age_ms: float,
        warn_age_ms: float,
    ) -> Dict[str, Any]:
        def _float_header(name: str) -> float:
            try:
                value = float(headers.get(name, "nan"))
            except (TypeError, ValueError):
                return float("nan")
            return value if np.isfinite(value) else float("nan")

        def _int_header(name: str) -> int:
            try:
                return int(headers.get(name, 0))
            except (TypeError, ValueError):
                return 0

        def _bool_header(name: str) -> bool:
            raw = str(headers.get(name, "")).strip().lower()
            return raw in {"1", "true", "yes"}

        fallback_reason = str(headers.get("X-AV-Selection-Fallback-Reason", "") or "")
        return {
            "payload": None,
            "consume_policy": str(headers.get("X-AV-Consume-Policy", consume_policy) or consume_policy),
            "freshness_budget_ms": float(max_age_ms),
            "warn_age_ms": float(warn_age_ms),
            "selection_result": str(headers.get("X-AV-Selection-Result", "") or ""),
            "selection_source": str(headers.get("X-AV-Selection-Source", "") or ""),
            "selected_age_ms": _float_header("X-AV-Selected-Payload-Age-Ms"),
            "selected_fresh": _bool_header("X-AV-Selected-Payload-Fresh"),
            "warn_age_exceeded": _bool_header("X-AV-Selected-Payload-Warn-Age-Exceeded"),
            "stale_drop_count": _int_header("X-AV-Stale-Drop-Count"),
            "drained_count": _int_header("X-AV-Drained-Count"),
            "max_drained_age_ms": _float_header("X-AV-Max-Drained-Age-Ms"),
            "server_queue_depth_after_select": _int_header(
                "X-AV-Server-Queue-Depth-After-Select"
            ),
            "server_oldest_age_ms_after_select": _float_header(
                "X-AV-Server-Oldest-Age-Ms-After-Select"
            ),
            "fallback_active": _bool_header("X-AV-Selection-Fallback-Active"),
            "fallback_reason_code": fallback_reason or PACKET_FALLBACK_PACKET_TIMEOUT,
            "join_failure_reason_code": str(headers.get("X-AV-Join-Failure-Reason", "") or ""),
            "join_failure_side_code": str(headers.get("X-AV-Join-Failure-Side", "") or ""),
            "selected_failure_contract_reason_code": str(
                headers.get("X-AV-Selected-Failure-Contract-Reason", "") or ""
            ),
            "selected_failure_source_stage_code": str(
                headers.get("X-AV-Selected-Failure-Source-Stage", "") or ""
            ),
            "source_bundle_close_reason": str(
                headers.get("X-AV-Source-Bundle-Close-Reason", "") or ""
            ),
            "source_bundle_deadline_ms": _float_header(
                "X-AV-Source-Bundle-Deadline-Ms"
            ),
            "source_bundle_age_ms": _float_header("X-AV-Source-Bundle-Age-Ms"),
            "source_bundle_inflight_count": _int_header(
                "X-AV-Source-Bundle-Inflight-Count"
            ),
            "source_bundle_vehicle_state_built": _bool_header(
                "X-AV-Source-Bundle-Vehicle-State-Built"
            ),
            "source_bundle_vehicle_state_enqueued": _bool_header(
                "X-AV-Source-Bundle-Vehicle-State-Enqueued"
            ),
            "source_bundle_vehicle_state_sent": _bool_header(
                "X-AV-Source-Bundle-Vehicle-State-Sent"
            ),
            "source_bundle_camera_requested": _bool_header(
                "X-AV-Source-Bundle-Camera-Requested"
            ),
            "source_bundle_camera_request_attempted": _bool_header(
                "X-AV-Source-Bundle-Camera-Request-Attempted"
            ),
            "source_bundle_camera_request_accepted": _bool_header(
                "X-AV-Source-Bundle-Camera-Request-Accepted"
            ),
            "source_bundle_camera_request_rejected_reason": str(
                headers.get("X-AV-Source-Bundle-Camera-Request-Rejected-Reason", "") or ""
            ),
            "source_bundle_camera_request_skipped_reason": str(
                headers.get("X-AV-Source-Bundle-Camera-Request-Skipped-Reason", "") or ""
            ),
            "source_bundle_camera_request_disposition_code": str(
                headers.get("X-AV-Source-Bundle-Camera-Request-Disposition-Code", "") or ""
            ),
            "source_bundle_camera_request_attempt_age_ms": _float_header(
                "X-AV-Source-Bundle-Camera-Request-Attempt-Age-Ms"
            ),
            "source_bundle_camera_request_accept_age_ms": _float_header(
                "X-AV-Source-Bundle-Camera-Request-Accept-Age-Ms"
            ),
            "source_bundle_camera_request_queue_depth": _int_header(
                "X-AV-Source-Bundle-Camera-Request-Queue-Depth"
            ),
            "source_bundle_active_transport_eligible": _bool_header(
                "X-AV-Source-Bundle-Active-Transport-Eligible"
            ),
            "source_bundle_debug_unbundled_capture": _bool_header(
                "X-AV-Source-Bundle-Debug-Unbundled-Capture"
            ),
            "camera_capture_contract_reason": str(
                headers.get("X-AV-Camera-Capture-Contract-Reason", "") or ""
            ),
            "source_bundle_camera_sent": _bool_header(
                "X-AV-Source-Bundle-Camera-Sent"
            ),
            "source_bundle_aborted_before_vehicle_send": _bool_header(
                "X-AV-Source-Bundle-Aborted-Before-Vehicle-Send"
            ),
            "source_bundle_abort_reason": str(
                headers.get("X-AV-Source-Bundle-Abort-Reason", "") or ""
            ),
            "source_bundle_vehicle_send_blocked_by_camera_request": _bool_header(
                "X-AV-Source-Bundle-Vehicle-Send-Blocked-By-Camera-Request"
            ),
            "source_bundle_superseded_before_send": _bool_header(
                "X-AV-Source-Bundle-Superseded-Before-Send"
            ),
            "source_key_present_camera": _bool_header("X-AV-Source-Key-Present-Camera"),
            "source_key_present_vehicle": _bool_header("X-AV-Source-Key-Present-Vehicle"),
            "timeout_event_delta": _int_header("X-AV-Timeout-Event-Delta"),
            "join_failure_event_count": _int_header("X-AV-Join-Failure-Event-Count"),
            "active_camera_excluded_event_delta": _int_header(
                "X-AV-Active-Camera-Excluded-Event-Delta"
            ),
            "active_camera_excluded_reason_code": str(
                headers.get("X-AV-Active-Camera-Excluded-Reason", "") or ""
            ),
            "unbundled_camera_entered_active_path_event_delta": _int_header(
                "X-AV-Unbundled-Camera-Entered-Active-Path-Event-Delta"
            ),
        }

    def get_next_sync_packet_payload_raw(self) -> Optional[Dict[str, Any]]:
        """Get the next synchronized packet payload (metadata + full vehicle state + raw image)."""
        try:
            response = self.session.get(f"{self.base_url}/api/sim/packet/next/raw", timeout=0.8)
            response.raise_for_status()
            return self._parse_sync_packet_payload_response(response)
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            return None
        except (requests.RequestException, ValueError):
            return None

    def get_latest_sync_packet_payload_raw(self) -> Optional[Dict[str, Any]]:
        """Get the latest synchronized packet payload without consuming the FIFO queue."""
        try:
            response = self.session.get(f"{self.base_url}/api/sim/packet/latest/raw", timeout=0.8)
            response.raise_for_status()
            return self._parse_sync_packet_payload_response(response)
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            return None
        except (requests.RequestException, ValueError):
            return None

    def get_fresh_sync_packet_payload_raw(
        self,
        *,
        consume_policy: str = PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
        max_age_ms: float = 250.0,
        warn_age_ms: float = 125.0,
        max_drain_count: int = 128,
        allow_stale_debug: bool = False,
    ) -> Dict[str, Any]:
        """Return one server-selected coherent payload with explicit server-side freshness accounting."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/sim/packet/next/raw",
                params={
                    "consume_policy": str(consume_policy),
                    "max_age_ms": float(max_age_ms),
                    "warn_age_ms": float(warn_age_ms),
                    "max_drain_count": int(max_drain_count),
                    "allow_stale_debug": bool(allow_stale_debug),
                },
                timeout=0.8,
            )
        except requests.exceptions.Timeout:
            return {
                "payload": None,
                "consume_policy": str(consume_policy),
                "freshness_budget_ms": float(max_age_ms),
                "warn_age_ms": float(warn_age_ms),
                "selection_result": "",
                "selection_source": "",
                "selected_age_ms": float("nan"),
                "selected_fresh": False,
                "warn_age_exceeded": False,
                "stale_drop_count": 0,
                "drained_count": 0,
                "max_drained_age_ms": float("nan"),
                "server_queue_depth_after_select": 0,
                "server_oldest_age_ms_after_select": float("nan"),
                "fallback_active": True,
                "fallback_reason_code": PACKET_FALLBACK_PACKET_TIMEOUT,
                "join_failure_reason_code": "",
                "join_failure_side_code": "",
                "selected_failure_contract_reason_code": "",
                "selected_failure_source_stage_code": "",
                "source_bundle_close_reason": "",
                "source_bundle_deadline_ms": float("nan"),
                "source_bundle_age_ms": float("nan"),
                "source_bundle_inflight_count": 0,
                "source_bundle_vehicle_state_built": False,
                "source_bundle_vehicle_state_enqueued": False,
                "source_bundle_vehicle_state_sent": False,
                "source_bundle_camera_requested": False,
                "source_bundle_camera_request_attempted": False,
                "source_bundle_camera_request_accepted": False,
                "source_bundle_camera_request_rejected_reason": "",
                "source_bundle_camera_request_skipped_reason": "",
                "source_bundle_camera_request_disposition_code": "",
                "source_bundle_camera_request_attempt_age_ms": float("nan"),
                "source_bundle_camera_request_accept_age_ms": float("nan"),
                "source_bundle_camera_request_queue_depth": 0,
                "source_bundle_active_transport_eligible": False,
                "source_bundle_debug_unbundled_capture": False,
                "camera_capture_contract_reason": "",
                "source_bundle_camera_sent": False,
                "source_bundle_aborted_before_vehicle_send": False,
                "source_bundle_abort_reason": "",
                "source_bundle_vehicle_send_blocked_by_camera_request": False,
                "source_bundle_superseded_before_send": False,
                "source_key_present_camera": False,
                "source_key_present_vehicle": False,
                "timeout_event_delta": 0,
                "join_failure_event_count": 0,
                "active_camera_excluded_event_delta": 0,
                "active_camera_excluded_reason_code": "",
                "unbundled_camera_entered_active_path_event_delta": 0,
            }
        except requests.RequestException:
            return {
                "payload": None,
                "consume_policy": str(consume_policy),
                "freshness_budget_ms": float(max_age_ms),
                "warn_age_ms": float(warn_age_ms),
                "selection_result": "",
                "selection_source": "",
                "selected_age_ms": float("nan"),
                "selected_fresh": False,
                "warn_age_exceeded": False,
                "stale_drop_count": 0,
                "drained_count": 0,
                "max_drained_age_ms": float("nan"),
                "server_queue_depth_after_select": 0,
                "server_oldest_age_ms_after_select": float("nan"),
                "fallback_active": True,
                "fallback_reason_code": PACKET_FALLBACK_PACKET_TIMEOUT,
                "join_failure_reason_code": "",
                "join_failure_side_code": "",
                "selected_failure_contract_reason_code": "",
                "selected_failure_source_stage_code": "",
                "source_bundle_close_reason": "",
                "source_bundle_deadline_ms": float("nan"),
                "source_bundle_age_ms": float("nan"),
                "source_bundle_inflight_count": 0,
                "source_bundle_vehicle_state_built": False,
                "source_bundle_vehicle_state_enqueued": False,
                "source_bundle_vehicle_state_sent": False,
                "source_bundle_camera_requested": False,
                "source_bundle_camera_request_attempted": False,
                "source_bundle_camera_request_accepted": False,
                "source_bundle_camera_request_rejected_reason": "",
                "source_bundle_camera_request_skipped_reason": "",
                "source_bundle_camera_request_disposition_code": "",
                "source_bundle_camera_request_attempt_age_ms": float("nan"),
                "source_bundle_camera_request_accept_age_ms": float("nan"),
                "source_bundle_camera_request_queue_depth": 0,
                "source_bundle_active_transport_eligible": False,
                "source_bundle_debug_unbundled_capture": False,
                "camera_capture_contract_reason": "",
                "source_bundle_camera_sent": False,
                "source_bundle_aborted_before_vehicle_send": False,
                "source_bundle_abort_reason": "",
                "source_bundle_vehicle_send_blocked_by_camera_request": False,
                "source_bundle_superseded_before_send": False,
                "source_key_present_camera": False,
                "source_key_present_vehicle": False,
                "timeout_event_delta": 0,
                "join_failure_event_count": 0,
                "active_camera_excluded_event_delta": 0,
                "active_camera_excluded_reason_code": "",
                "unbundled_camera_entered_active_path_event_delta": 0,
            }

        if response.status_code == 404:
            return self._parse_selection_headers(
                response.headers,
                consume_policy=str(consume_policy),
                max_age_ms=float(max_age_ms),
                warn_age_ms=float(warn_age_ms),
            )

        response.raise_for_status()
        payload = self._parse_sync_packet_payload_response(response)
        packet_meta = payload.get("packet_meta") or {}
        fallback_reason = str(
            packet_meta.get("selection_fallback_reason_code")
            or packet_meta.get("fallback_reason_code")
            or PACKET_FALLBACK_NONE
        )
        return {
            "payload": payload,
            "consume_policy": str(packet_meta.get("consume_policy") or consume_policy),
            "freshness_budget_ms": float(max_age_ms),
            "warn_age_ms": float(warn_age_ms),
            "selection_result": str(packet_meta.get("selection_result") or ""),
            "selection_source": str(packet_meta.get("payload_selection_source") or ""),
            "selected_age_ms": float(
                packet_meta.get(
                    "payload_selected_age_ms",
                    self._extract_sync_packet_age_ms(payload),
                )
            ),
            "selected_fresh": bool(packet_meta.get("payload_selected_fresh", True)),
            "warn_age_exceeded": bool(
                packet_meta.get("payload_warn_age_exceeded", False)
            ),
            "stale_drop_count": int(packet_meta.get("payload_stale_drop_count", 0) or 0),
            "drained_count": int(packet_meta.get("payload_drained_count", 0) or 0),
            "max_drained_age_ms": float(
                packet_meta.get("payload_max_drained_age_ms", float("nan"))
            ),
            "server_queue_depth_after_select": int(
                packet_meta.get("payload_server_queue_depth_after_select", 0) or 0
            ),
            "server_oldest_age_ms_after_select": float(
                packet_meta.get(
                    "payload_server_oldest_age_ms_after_select",
                    float("nan"),
                )
            ),
            "fallback_active": bool(
                packet_meta.get("selection_fallback_active", False)
            ),
            "fallback_reason_code": fallback_reason,
            "join_failure_reason_code": str(
                packet_meta.get("join_failure_reason_code") or ""
            ),
            "join_failure_side_code": str(
                packet_meta.get("join_failure_side_code") or ""
            ),
            "selected_failure_contract_reason_code": str(
                packet_meta.get("selected_failure_contract_reason_code") or ""
            ),
            "selected_failure_source_stage_code": str(
                packet_meta.get("selected_failure_source_stage_code") or ""
            ),
            "source_bundle_close_reason": str(
                packet_meta.get("source_bundle_close_reason") or ""
            ),
            "source_bundle_deadline_ms": float(
                packet_meta.get("source_bundle_deadline_ms", float("nan"))
            ),
            "source_bundle_age_ms": float(
                packet_meta.get("source_bundle_age_ms", float("nan"))
            ),
            "source_bundle_inflight_count": int(
                packet_meta.get("source_bundle_inflight_count", 0) or 0
            ),
            "source_bundle_vehicle_state_built": packet_meta.get(
                "source_bundle_vehicle_state_built"
            ),
            "source_bundle_vehicle_state_enqueued": packet_meta.get(
                "source_bundle_vehicle_state_enqueued"
            ),
            "source_bundle_vehicle_state_sent": packet_meta.get(
                "source_bundle_vehicle_state_sent"
            ),
            "source_bundle_camera_requested": packet_meta.get(
                "source_bundle_camera_requested"
            ),
            "source_bundle_camera_request_attempted": packet_meta.get(
                "source_bundle_camera_request_attempted"
            ),
            "source_bundle_camera_request_accepted": packet_meta.get(
                "source_bundle_camera_request_accepted"
            ),
            "source_bundle_camera_request_rejected_reason": str(
                packet_meta.get("source_bundle_camera_request_rejected_reason") or ""
            ),
            "source_bundle_camera_request_skipped_reason": str(
                packet_meta.get("source_bundle_camera_request_skipped_reason") or ""
            ),
            "source_bundle_camera_request_disposition_code": str(
                packet_meta.get("source_bundle_camera_request_disposition_code") or ""
            ),
            "source_bundle_camera_request_attempt_age_ms": float(
                packet_meta.get("source_bundle_camera_request_attempt_age_ms", float("nan"))
            ),
            "source_bundle_camera_request_accept_age_ms": float(
                packet_meta.get("source_bundle_camera_request_accept_age_ms", float("nan"))
            ),
            "source_bundle_camera_request_queue_depth": int(
                packet_meta.get("source_bundle_camera_request_queue_depth", 0) or 0
            ),
            "source_bundle_active_transport_eligible": bool(
                packet_meta.get("source_bundle_active_transport_eligible", False)
            ),
            "source_bundle_debug_unbundled_capture": bool(
                packet_meta.get("source_bundle_debug_unbundled_capture", False)
            ),
            "camera_capture_contract_reason": str(
                packet_meta.get("camera_capture_contract_reason") or ""
            ),
            "source_bundle_camera_sent": packet_meta.get(
                "source_bundle_camera_sent"
            ),
            "source_bundle_aborted_before_vehicle_send": packet_meta.get(
                "source_bundle_aborted_before_vehicle_send"
            ),
            "source_bundle_abort_reason": str(
                packet_meta.get("source_bundle_abort_reason") or ""
            ),
            "source_bundle_vehicle_send_blocked_by_camera_request": packet_meta.get(
                "source_bundle_vehicle_send_blocked_by_camera_request"
            ),
            "source_bundle_superseded_before_send": packet_meta.get(
                "source_bundle_superseded_before_send"
            ),
            "source_key_present_camera": bool(
                packet_meta.get("source_key_present_camera", False)
            ),
            "source_key_present_vehicle": bool(
                packet_meta.get("source_key_present_vehicle", False)
            ),
            "timeout_event_delta": int(
                packet_meta.get("timeout_event_delta", 0) or 0
            ),
            "join_failure_event_count": int(
                packet_meta.get("join_failure_event_count", 0) or 0
            ),
            "active_camera_excluded_event_delta": int(
                packet_meta.get("active_camera_excluded_event_delta", 0) or 0
            ),
            "active_camera_excluded_reason_code": str(
                packet_meta.get("active_camera_excluded_reason_code") or ""
            ),
            "unbundled_camera_entered_active_path_event_delta": int(
                packet_meta.get(
                    "unbundled_camera_entered_active_path_event_delta", 0
                )
                or 0
            ),
        }
    
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
        try:
            self._parallel_executor.shutdown(wait=False)
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
