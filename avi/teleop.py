
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

from arm import Arm
from camera import Camera

HOST = "0.0.0.0"
PORT = 8000
STEP = 5
HOME = (90, 90)

arm = None
arm_lock = threading.Lock()

_latest_jpeg = None
_frame_lock = threading.Lock()
_camera_running = True


def camera_loop():
    global _latest_jpeg
    cam = Camera()
    while _camera_running:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            with _frame_lock:
                _latest_jpeg = buf.tobytes()
        time.sleep(0.03)
    cam.release()


def do_move(direction):
    dx = dy = 0
    if direction == "a":      dx = -STEP
    elif direction == "d":    dx = STEP
    elif direction == "w":    dy = -STEP
    elif direction == "s":    dy = STEP
    elif direction == "center":
        with arm_lock:
            dx = arm.base - HOME[0]
            dy = arm.tilt - HOME[1]
    with arm_lock:
        try:
            if dx or dy:
                arm.update(dx, dy)
            return arm.base, arm.tilt, arm.connection_healthy
        except Exception as e:
            print("[TELEOP] move failed:", e)
            return arm.base, arm.tilt, False


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Arm Teleop</title>
<style>
  body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;
       display:flex;flex-direction:column;align-items:center}
  h1{font-size:16px;font-weight:600;margin:10px}
  #vid{max-width:100%;border:2px solid #333;background:#000}
  #hud{display:flex;gap:24px;margin:10px;font-variant-numeric:tabular-nums}
  .k{display:inline-block;min-width:2.4em;text-align:center;padding:4px 8px;
     margin:2px;border:1px solid #555;border-radius:6px;background:#222}
  .k.on{background:#2d6cdf;border-color:#2d6cdf}
  .row{text-align:center}
  small{color:#999}
</style></head>
<body>
  <h1>Arm Teleop &mdash; hold W / A / S / D to drive</h1>
  <img id="vid" src="/video" alt="live feed">
  <div id="hud">
    <div>base: <b id="base">--</b>&deg;</div>
    <div>tilt: <b id="tilt">--</b>&deg;</div>
    <div id="health">link ok</div>
  </div>
  <div id="keys">
    <div class="row"><span class="k" id="kw">W</span></div>
    <div class="row">
      <span class="k" id="ka">A</span>
      <span class="k" id="ks">S</span>
      <span class="k" id="kd">D</span>
    </div>
    <div class="row"><small>Space = stop &nbsp;|&nbsp; C = center</small></div>
  </div>
<script>
const held = {};           // which keys are currently down
const timers = {};         // repeat interval per key
const MAP = {w:'w',a:'a',s:'s',d:'d'};
const REPEAT_MS = 120;

function setKeyUI(k, on){
  const el = document.getElementById('k'+k);
  if (el) el.classList.toggle('on', on);
}
function updateHud(j){
  document.getElementById('base').textContent = j.base;
  document.getElementById('tilt').textContent = j.tilt;
  const h = document.getElementById('health');
  if (j.healthy){ h.textContent='link ok'; h.classList.remove('bad'); }
  else { h.textContent='LINK UNHEALTHY'; h.classList.add('bad'); }
}
async function move(dir){
  try{
    const r = await fetch('/move?dir='+dir);
    updateHud(await r.json());
  }catch(e){ /* transient network blip; next tick retries */ }
}
function startKey(k){
  if (held[k]) return;         // ignore OS auto-repeat
  held[k] = true; setKeyUI(k, true);
  move(k);                     // immediate first step
  timers[k] = setInterval(()=>move(k), REPEAT_MS);
}
function stopKey(k){
  if (!held[k]) return;
  held[k] = false; setKeyUI(k, false);
  clearInterval(timers[k]);
}
function stopAll(){ for (const k in MAP) stopKey(k); }

window.addEventListener('keydown', e=>{
  const k = e.key.toLowerCase();
  if (MAP[k]){ e.preventDefault(); startKey(k); }
  else if (k === ' '){ e.preventDefault(); stopAll(); }
  else if (k === 'c'){ e.preventDefault(); move('center'); }
});
window.addEventListener('keyup', e=>{
  const k = e.key.toLowerCase();
  if (MAP[k]){ e.preventDefault(); stopKey(k); }
});
// safety: if the tab loses focus, stop driving immediately
window.addEventListener('blur', stopAll);
// prime the HUD
move('center');
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _frame_lock:
                        jpg = _latest_jpeg
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        ("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif parsed.path == "/move":
            q = parse_qs(parsed.query)
            direction = (q.get("dir", [""])[0]).lower()
            base, tilt, healthy = do_move(direction)
            payload = ('{"base":%d,"tilt":%d,"healthy":%s}'
                       % (base, tilt, "true" if healthy else "false")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_response(404)
            self.end_headers()


def main():
    global arm
    print("Starting camera thread (settling auto-exposure)...")
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    time.sleep(2.0)

    print("Opening arm on COM5...")
    arm = Arm()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\nTeleop server up. On the MacBook, open:")
    print(f"    http://<laptop-tailscale-ip>:{PORT}/")
    print(f"(run `tailscale ip -4` on this laptop to get the IP)")
    print("Ctrl+C here to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        global _camera_running
        _camera_running = False
        server.server_close()


if __name__ == "__main__":
    main()
