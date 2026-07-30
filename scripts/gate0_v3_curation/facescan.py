"""Per-frame face timeline over the whole film, 2 fps, parallel across cores.

Asymmetric detection protocol, deliberately:
  STRICT      (high precision) -> used to certify a face IS present. False positives here would put a
              non-face clip into the FACE condition.
  PERMISSIVE  (high recall)    -> used to certify a face is ABSENT. False negatives here would put a
              face into the face-ABSENT baseline, which is exactly the defect found in NONFACE_03 and
              NONFACE_14. So absence is only claimed when the most sensitive settings find nothing.

Emits, per sampled frame: counts and max bbox area fraction for each detector/setting, so any windowing
or dominance rule can be applied later without re-scanning.
"""
import os
import subprocess
import sys

import cv2
import numpy as np

# one process per core: stop OpenCV from oversubscribing 16 cores x 14 workers
cv2.setNumThreads(1)

W, H = 480, 270
FPS = 2.0
FF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg')
FILM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charade.mp4')

CASC = {k: cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, f'haarcascade_{k}.xml'))
        for k in ('frontalface_default', 'frontalface_alt2', 'profileface')}

STRICT = dict(scaleFactor=1.08, minNeighbors=6, minSize=(40, 40))
# high recall, but not absurdly so: a 28px face on a 480x270 frame is ~10% of frame height,
# well below any 'dominant' threshold, so this still certifies absence of anything visible.
PERMISSIVE = dict(scaleFactor=1.06, minNeighbors=2, minSize=(28, 28))


def boxes(gray, casc, params):
    f = CASC[casc].detectMultiScale(gray, **params)
    if len(f) == 0:
        return 0, 0.0, 0.0
    areas = [(w * h) / float(W * H) for _, _, w, h in f]
    return len(f), max(areas), sum(areas)


def scan(start, dur):
    cmd = [FF, '-v', 'error', '-ss', str(start), '-t', str(dur), '-i', FILM,
           '-vf', f'fps={FPS},scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (W * H)
    rows = []
    for i in range(n):
        g = np.frombuffer(raw[i * W * H:(i + 1) * W * H], dtype=np.uint8).reshape(H, W)
        g = cv2.equalizeHist(g)
        sn, sa, st = boxes(g, 'frontalface_default', STRICT)
        s2n, s2a, _ = boxes(g, 'frontalface_alt2', STRICT)
        pn, pa, pt = boxes(g, 'frontalface_default', PERMISSIVE)
        p2n, p2a, _ = boxes(g, 'frontalface_alt2', PERMISSIVE)
        prn, pra, _ = boxes(g, 'profileface', PERMISSIVE)
        rows.append((start + i / FPS,
                     sn, sa, st, s2n, s2a,          # strict frontal (two cascades)
                     pn, pa, p2n, p2a,              # permissive frontal (two cascades)
                     prn, pra))                     # permissive profile
    return np.array(rows, dtype=np.float32)


if __name__ == '__main__':
    start, dur, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    a = scan(start, dur)
    np.save(out, a)
    print(f'{out}: {len(a)} frames  {start:.0f}..{start + dur:.0f}s')
