"""Live track-perception test on the robot. Never requests motion.

Ported from the PC "Camera Test" prototype (track_test.py + hsv_tuner.py).
The Jetson has no display, so instead of OpenCV windows this node:

  * logs found / error / blue fractions twice a second,
  * publishes the debug view (camera overlay | masks) on
    perception/debug/compressed for RViz on your PC,
  * serves the same view with HSV sliders at http://ROBOT_IP:8081/
    (set web_port to 0 to turn this off).

HSV ranges are ROS parameters (yellow_lo, yellow_hi, blue_lo, blue_hi), so
they can also be changed live with `ros2 param set`. On startup they are
overridden by the JSON file at hsv_config_path if it exists; the file uses the
same format as the PC prototype's hsv_config.json.

Services (std_srvs/srv/Trigger), relative to the robot namespace:
  perception/save_config    write the current HSV ranges to hsv_config_path
  perception/reload_config  re-read hsv_config_path
  perception/save_snapshot  save the latest full-resolution camera frame
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger

from team_navigation.perception import (
    CONFIG_KEYS, DEFAULT_CONFIG, Perception, draw_debug, load_config,
    save_config, validate_triplet)


class TrackTest(Node):
    def __init__(self):
        super().__init__('track_test')
        p = self.declare_parameter
        self.config_path = p('hsv_config_path', '~/hsv_config.json').value
        self.snapshot_dir = p('snapshot_dir', '~/snapshots').value
        self.base_speed = p('preview_base_speed', 0.5).value
        self.k_p = p('preview_k_p', 0.5).value
        self.stale_after = p('stale_after_s', 1.0).value
        web_port = p('web_port', 8081).value
        for key in CONFIG_KEYS:
            p(key, DEFAULT_CONFIG[key])

        self.lock = threading.RLock()          # guards perception config
        self.frame_cond = threading.Condition()  # guards latest_* below
        self.latest_debug_jpeg = None
        self.latest_camera_jpeg = None
        self.frame_seq = 0
        self.web_clients = 0

        self.perception = Perception(self.current_params())
        self.add_on_set_parameters_callback(self.on_set_parameters)
        ok, message = self.reload_config()
        (self.get_logger().info if ok else self.get_logger().warn)(message)

        self.fps, self.last_t, self.last_print = 0.0, time.monotonic(), 0.0
        self.start_t = time.monotonic()
        self.last_frame_t = None
        self.stale_warned = False

        self.debug_pub = self.create_publisher(
            CompressedImage, 'perception/debug/compressed', qos_profile_sensor_data)
        self.create_subscription(
            CompressedImage, 'camera/image_raw/compressed', self.image_cb,
            qos_profile_sensor_data)
        for name, fn in (('save_config', self.save_config),
                         ('reload_config', self.reload_config),
                         ('save_snapshot', self.save_snapshot)):
            self.create_service(Trigger, f'perception/{name}', self._trigger(fn))
        self.create_timer(1.0, self.check_stale)

        self.web = None
        if web_port:
            try:
                self.web = start_web_server(self, web_port)
                self.get_logger().info(f'Debug view and HSV tuner: http://ROBOT_IP:{web_port}/')
            except OSError as exc:
                self.get_logger().error(f'Web view disabled; port {web_port}: {exc}')
        self.get_logger().info('Waiting for camera/image_raw/compressed '
                               '(is the a2r_robot base launch running?)')

    # ------------------------------------------------------------ config
    def current_params(self):
        return {k: list(self.get_parameter(k).value) for k in CONFIG_KEYS}

    def on_set_parameters(self, params):
        updates = {}
        for param in params:
            if param.name in CONFIG_KEYS:
                try:
                    updates[param.name] = validate_triplet(param.value)
                except (TypeError, ValueError) as exc:
                    return SetParametersResult(successful=False, reason=f'{param.name}: {exc}')
            elif param.name == 'preview_base_speed':
                self.base_speed = float(param.value)
            elif param.name == 'preview_k_p':
                self.k_p = float(param.value)
        if updates:
            with self.lock:
                cfg = dict(self.perception.cfg, **updates)
                self.perception.set_config(cfg)
        return SetParametersResult(successful=True)

    def set_hsv(self, updates):
        """Apply HSV updates through the parameter system (keeps ros2 param in sync)."""
        params = [Parameter(k, Parameter.Type.INTEGER_ARRAY, validate_triplet(v))
                  for k, v in updates.items() if k in CONFIG_KEYS]
        for result in self.set_parameters(params):
            if not result.successful:
                return False, result.reason
        return True, 'updated'

    def get_hsv(self):
        with self.lock:
            return {k: list(v) for k, v in self.perception.cfg.items()}

    def save_config(self):
        try:
            path = save_config(self.get_hsv(), self.config_path)
        except OSError as exc:
            return False, f'Could not save HSV config: {exc}'
        return True, f'Saved HSV config to {path}: {self.get_hsv()}'

    def reload_config(self):
        path = os.path.expanduser(self.config_path)
        if not os.path.exists(path):
            return True, f'No {path}; using HSV parameters {self.get_hsv()}'
        try:
            ok, message = self.set_hsv(load_config(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            ok, message = False, str(exc)
        if not ok:
            return False, f'Could not load {path}: {message}'
        return True, f'Loaded HSV config from {path}: {self.get_hsv()}'

    def save_snapshot(self):
        with self.frame_cond:
            jpeg = self.latest_camera_jpeg
        if jpeg is None:
            return False, 'No camera frame received yet'
        directory = os.path.expanduser(self.snapshot_dir)
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            with open(path, 'wb') as f:
                f.write(jpeg)  # the camera's own JPEG, full resolution, no re-encode
        except OSError as exc:
            return False, f'Could not save snapshot: {exc}'
        return True, f'Saved {path}'

    def _trigger(self, fn):
        def handle(_request, response):
            response.success, response.message = fn()
            (self.get_logger().info if response.success else self.get_logger().error)(
                response.message)
            return response
        return handle

    # ------------------------------------------------------------ images
    def image_cb(self, msg):
        now = time.monotonic()
        self.last_frame_t = now
        if self.stale_warned:
            self.get_logger().info('Camera frames resumed')
            self.stale_warned = False
        jpeg = bytes(msg.data)
        # Decoding at half size (640x360 from 1280x720) is much faster and
        # still larger than the 320x240 processing resolution.
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        if frame is None:
            self.get_logger().warn('Could not decode camera frame', throttle_duration_sec=5.0)
            return
        with self.lock:
            r = self.perception.process(frame)

        print(frame)

        # What a P-controller would send to the treads (preview only; this
        # node publishes no motion request).
        left_cmd = float(np.clip(self.base_speed + self.k_p * r.error, -1, 1))
        right_cmd = float(np.clip(self.base_speed - self.k_p * r.error, -1, 1))

        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.last_t, 1e-6))
        self.last_t = now
        if now - self.last_print > 0.5:
            self.get_logger().info(
                f'found={r.found!s:5}  error={r.error:+.2f}  yellow_px={r.yellow_pixels:5d}  '
                f'blueL={r.blue_left:.2f} blueR={r.blue_right:.2f}  warn={r.blue_warning}  '
                f'{self.fps:.0f} fps')
            self.last_print = now

        want_debug = self.web_clients > 0 or self.debug_pub.get_subscription_count() > 0
        debug_jpeg = None
        if want_debug:
            ok, buf = cv2.imencode('.jpg', draw_debug(r, left_cmd, right_cmd, self.fps),
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                debug_jpeg = buf.tobytes()
                out = CompressedImage()
                out.header = msg.header
                out.format = 'jpeg'
                out.data = debug_jpeg
                self.debug_pub.publish(out)
        with self.frame_cond:
            self.latest_camera_jpeg = jpeg
            if debug_jpeg is not None:
                self.latest_debug_jpeg = debug_jpeg
                self.frame_seq += 1
                self.frame_cond.notify_all()

    def check_stale(self):
        if self.stale_warned:
            return
        age = time.monotonic() - (self.last_frame_t or self.start_t)
        if age > max(self.stale_after, 3.0 if self.last_frame_t is None else 0.0):
            self.get_logger().warn(
                f'No camera frame for {age:.1f} s. Is the base launch running, and was '
                'rvr-camera.service stopped before it started?')
            self.stale_warned = True

    def destroy_node(self):
        if self.web is not None:
            self.web.shutdown()
            self.web.server_close()
        super().destroy_node()


# ---------------------------------------------------------------- web view
PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Track test</title>
<style>
body{margin:0;background:#10151d;color:#e6edf3;font:15px system-ui}
main{max-width:1280px;margin:auto;padding:12px 16px}
h1{font-size:19px;margin:4px 0 10px}img{width:100%;display:block;background:#080b10}
.row{display:flex;flex-wrap:wrap;gap:8px 24px;align-items:center;margin:12px 0}
label{display:grid;grid-template-columns:52px 1fr 36px;gap:8px;align-items:center;min-width:260px;flex:1}
input[type=range]{width:100%}button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid #3a4656;
background:#1c2530;color:inherit;cursor:pointer}button.on{background:#2f6fb3;border-color:#2f6fb3}
#msg{color:#9caebf;font-size:13px;min-height:1.2em}
</style>
<main>
<h1>Track test</h1>
<img id="feed" src="/stream.mjpg" alt="Camera overlay and colour masks">
<div class="row"><span>Tuning:</span>
<button id="yellow" class="on" onclick="pick('yellow')">Yellow</button>
<button id="blue" onclick="pick('blue')">Blue</button>
<span style="flex:1"></span>
<button onclick="act('save')">Save to file</button>
<button onclick="act('reload')">Reload file</button>
<button onclick="act('snapshot')">Snapshot</button></div>
<div class="row" id="sliders"></div>
<div id="msg"></div>
</main>
<script>
const S=[['H min','lo',0,179],['S min','lo',1,255],['V min','lo',2,255],
         ['H max','hi',0,179],['S max','hi',1,255],['V max','hi',2,255]];
let cfg={},color='yellow',timer=null;
const box=document.getElementById('sliders'),msg=document.getElementById('msg');
S.forEach(([name,,,max],i)=>{box.insertAdjacentHTML('beforeend',
 `<label>${name}<input type=range min=0 max=${max} id=s${i}><output id=o${i}></output></label>`);
 document.getElementById('s'+i).oninput=send;});
function show(){S.forEach(([,end,idx],i)=>{const v=cfg[color+'_'+end][idx];
 document.getElementById('s'+i).value=v;document.getElementById('o'+i).value=v;});}
function pick(c){color=c;['yellow','blue'].forEach(k=>document.getElementById(k).classList.toggle('on',k==c));show();}
function send(){const lo=[0,1,2].map(i=>+document.getElementById('s'+i).value),
 hi=[3,4,5].map(i=>+document.getElementById('s'+i).value);
 cfg[color+'_lo']=lo;cfg[color+'_hi']=hi;S.forEach((_,i)=>document.getElementById('o'+i).value=document.getElementById('s'+i).value);
 clearTimeout(timer);timer=setTimeout(()=>post('/config',{[color+'_lo']:lo,[color+'_hi']:hi}),80);}
async function post(path,body){try{const r=await fetch(path,{method:'POST',body:JSON.stringify(body||{})});
 const j=await r.json();msg.textContent=j.message;if(j.config){cfg=j.config;show();}}catch(e){msg.textContent='Robot unreachable: '+e;}}
function act(a){post('/'+a);}
fetch('/config').then(r=>r.json()).then(j=>{cfg=j.config;show();});
const feed=document.getElementById('feed');
feed.onerror=()=>setTimeout(()=>{feed.src='/stream.mjpg?t='+Date.now()},1500);
</script></html>"""


def start_web_server(node, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the ROS log readable
            pass

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                body = PAGE.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == '/config':
                self._json({'config': node.get_hsv()})
            elif path == '/stream.mjpg':
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(length) or b'{}')
            except ValueError:
                return self._json({'message': 'bad JSON'}, 400)
            actions = {'/save': node.save_config, '/reload': node.reload_config,
                       '/snapshot': node.save_snapshot}
            if self.path == '/config':
                try:
                    ok, message = node.set_hsv(body)
                except (TypeError, ValueError) as exc:
                    ok, message = False, str(exc)
            elif self.path in actions:
                ok, message = actions[self.path]()
                (node.get_logger().info if ok else node.get_logger().error)(message)
            else:
                return self.send_error(404)
            self._json({'ok': ok, 'message': message, 'config': node.get_hsv()},
                       200 if ok else 400)

        def _stream(self):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with node.frame_cond:
                node.web_clients += 1  # tells image_cb to start drawing the preview
                seq = node.frame_seq
            try:
                while rclpy.ok():
                    with node.frame_cond:
                        node.frame_cond.wait_for(lambda: node.frame_seq != seq, timeout=1.0)
                        if node.frame_seq == seq or node.latest_debug_jpeg is None:
                            continue
                        jpeg, seq = node.latest_debug_jpeg, node.frame_seq
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                     + f'Content-Length: {len(jpeg)}\r\n\r\n'.encode()
                                     + jpeg + b'\r\n')
            except OSError:  # browser closed the tab or the connection dropped
                pass
            finally:
                with node.frame_cond:
                    node.web_clients -= 1

    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    rclpy.init()
    node = TrackTest()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
