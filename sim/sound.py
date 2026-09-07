"""Sound propagation: Eikonal arrival-time fields over the tile grid.

A field is solved once per sound event from its origin. Every listener then
samples that one field: arrival time tells you when they hear it, and the
negative gradient tells you which direction they think it came from. The
bearing is a property of the route the sound actually took, so a listener
behind a wall gets a bearing pointing at the doorway rather than at the source.

Two backends:

  "skfmm"  scikit-fmm, C-backed fast marching. ~2.4 ms for a full 165x110
           field. Accurate to well under a degree of bearing. Only ships
           Windows wheels, up to cp313, so on Python 3.14 or on Linux/macOS
           it needs a source build.

  "python" pure-Python heapq + Eikonal update. No dependencies. ~1.75 ms for
           a footstep-sized field, ~21 ms for a full one. Same accuracy as
           skfmm; just slower.

Both are exact enough for bearings. skimage's MCP_Geometric is faster to
install but its 8-connected metric quantises bearings to 22.5 degree steps
(up to 19 degrees of error), which defeats the point, so it is not offered.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np

try:
    import skfmm as _skfmm
except ImportError:
    _skfmm = None

try:
    from numba import njit as _njit
except ImportError:
    _njit = None

if _njit is not None:
    BACKEND = "numba"
elif _skfmm is not None:
    BACKEND = "skfmm"
else:
    BACKEND = "python"


def backend_report() -> str:
    if BACKEND == "numba":
        return "sound backend: numba (JIT-compiled, ~25x the pure solver)"
    if BACKEND == "skfmm":
        return "sound backend: skfmm (C fast marching)"
    return ("sound backend: python (pure) \u2014 pip install numba for a "
            "25x speedup with no compiler needed")


if _njit is not None:

    @_njit(cache=True, fastmath=False)
    def _solve_numba(cost, ox, oy, energy):
        """Same algorithm as _solve_python, compiled.

        heapq and Python lists do not compile, so the heap is two
        preallocated arrays with hand-written sift-up/sift-down, and the
        working arrays are typed numpy. Output matches the pure solver to
        within ~1e-5.
        """
        h, w = cost.shape
        n = w * h
        t = np.full(n, np.inf)
        tr = np.full(n, np.inf)
        done = np.zeros(n, np.uint8)
        hv = np.empty(n * 4)
        hi = np.empty(n * 4, np.int64)
        hn = 0

        s0 = oy * w + ox
        t[s0] = 0.0
        tr[s0] = 0.0
        hv[0] = 0.0
        hi[0] = s0
        hn = 1

        while hn > 0:
            tv = hv[0]
            i = hi[0]
            hn -= 1
            hv[0] = hv[hn]
            hi[0] = hi[hn]
            c = 0
            while True:
                l = 2 * c + 1
                r = l + 1
                mn = c
                if l < hn and hv[l] < hv[mn]:
                    mn = l
                if r < hn and hv[r] < hv[mn]:
                    mn = r
                if mn == c:
                    break
                hv[c], hv[mn] = hv[mn], hv[c]
                hi[c], hi[mn] = hi[mn], hi[c]
                c = mn

            if done[i]:
                continue
            if tv > energy:
                break
            done[i] = 1
            x = i % w
            y = i // w

            for k in range(4):
                if k == 0:
                    if x == 0:
                        continue
                    j = i - 1
                elif k == 1:
                    if x == w - 1:
                        continue
                    j = i + 1
                elif k == 2:
                    if y == 0:
                        continue
                    j = i - w
                else:
                    if y == h - 1:
                        continue
                    j = i + w
                if done[j]:
                    continue
                f = cost[j // w, j % w]
                if not np.isfinite(f):
                    continue
                nx = j % w
                ny = j // w

                a = np.inf
                ta = np.inf
                if nx > 0 and t[j - 1] < a:
                    a = t[j - 1]
                    ta = tr[j - 1]
                if nx < w - 1 and t[j + 1] < a:
                    a = t[j + 1]
                    ta = tr[j + 1]
                b = np.inf
                tb = np.inf
                if ny > 0 and t[j - w] < b:
                    b = t[j - w]
                    tb = tr[j - w]
                if ny < h - 1 and t[j + w] < b:
                    b = t[j + w]
                    tb = tr[j + w]

                if a == np.inf and b == np.inf:
                    continue
                if a == np.inf:
                    nt = b + f
                    ntr = tb + 1.0
                elif b == np.inf:
                    nt = a + f
                    ntr = ta + 1.0
                else:
                    d = a - b
                    if -f < d < f:
                        nt = (a + b + math.sqrt(2.0 * f * f - d * d)) * 0.5
                        dt = ta - tb
                        if -1.0 < dt < 1.0:
                            ntr = (ta + tb + math.sqrt(2.0 - dt * dt)) * 0.5
                        else:
                            ntr = min(ta, tb) + 1.0
                    elif a < b:
                        nt = a + f
                        ntr = ta + 1.0
                    else:
                        nt = b + f
                        ntr = tb + 1.0

                if nt < t[j] - 1e-6:
                    t[j] = nt
                    tr[j] = ntr
                    hv[hn] = nt
                    hi[hn] = j
                    c = hn
                    hn += 1
                    while c > 0:
                        pa = (c - 1) // 2
                        if hv[pa] <= hv[c]:
                            break
                        hv[pa], hv[c] = hv[c], hv[pa]
                        hi[pa], hi[c] = hi[c], hi[pa]
                        c = pa

        for q in range(n):
            if t[q] > energy:
                t[q] = np.inf
                tr[q] = np.inf
        return t.reshape(h, w), tr.reshape(h, w)


def warmup() -> float:
    """Trigger JIT compilation up front. Returns seconds spent."""
    if _njit is None:
        return 0.0
    t0 = time.perf_counter()
    tiny = np.ones((8, 8), dtype=np.float64)
    _solve_numba(tiny, 4, 4, 4.0)
    return time.perf_counter() - t0


def _solve_python(cost: np.ndarray, ox: int, oy: int, max_t: float) -> np.ndarray:
    """Dijkstra with the Eikonal relaxation, early-terminated at max_t."""
    h, w = cost.shape
    INF = math.inf
    t = [INF] * (w * h)
    done = bytearray(w * h)
    flat = cost.ravel()

    start = oy * w + ox
    t[start] = 0.0
    heap = [(0.0, start)]
    push, pop = heapq.heappush, heapq.heappop

    while heap:
        tv, i = pop(heap)
        if done[i]:
            continue
        if tv > max_t:
            break
        done[i] = 1
        x = i % w
        y = i // w

        for j in ((i - 1) if x > 0 else -1,
                  (i + 1) if x < w - 1 else -1,
                  (i - w) if y > 0 else -1,
                  (i + w) if y < h - 1 else -1):
            if j < 0 or done[j]:
                continue
            f = flat[j]
            if not math.isfinite(f):
                continue
            nx = j % w
            ny = j // w

            a = INF
            if nx > 0:
                a = min(a, t[j - 1])
            if nx < w - 1:
                a = min(a, t[j + 1])
            b = INF
            if ny > 0:
                b = min(b, t[j - w])
            if ny < h - 1:
                b = min(b, t[j + w])

            if a == INF and b == INF:
                continue
            if a == INF:
                nt = b + f
            elif b == INF:
                nt = a + f
            else:
                d = a - b
                if -f < d < f:
                    nt = (a + b + math.sqrt(2.0 * f * f - d * d)) * 0.5
                else:
                    nt = min(a, b) + f

            if nt < t[j] - 1e-6:
                t[j] = nt
                push(heap, (nt, j))

    arr = np.asarray(t, dtype=np.float32).reshape(h, w)
    arr[arr > max_t] = np.inf
    return arr


def _solve_skfmm(cost: np.ndarray, ox: int, oy: int, max_t: float) -> np.ndarray:
    phi = np.ones(cost.shape, dtype=np.float64)
    phi[oy, ox] = -1.0
    blocked = ~np.isfinite(cost)
    speed = np.where(blocked, 1.0, 1.0 / np.maximum(cost, 1e-6))
    out = _skfmm.travel_time(np.ma.MaskedArray(phi, blocked), speed)
    arr = np.ma.filled(out, np.inf).astype(np.float32)
    arr[arr > max_t] = np.inf
    return arr


@dataclass
class SoundField:
    """One sound event, on a cropped window.

    Two separate quantities, because they are separate physics:

      `t`       accumulated ATTENUATION along the least-lossy route. This is
                what decides whether a listener hears the sound at all and how
                confident their bearing is. Walls are expensive here.
      `travel`  the geometric LENGTH of that same route, in cells. This is
                when the sound arrives. Walls are not expensive here - sound
                through a wall is quiet, not slow.

    Conflating the two is the obvious shortcut and it is wrong: it makes a
    sealed room next door light up seconds after a room further away that
    happens to have a door, when physically both arrive at nearly the same
    moment and only differ in loudness.
    """

    t: np.ndarray
    travel: np.ndarray
    x0: int
    y0: int
    origin: tuple[int, int]
    energy: float
    backend: str
    max_travel: float = 0.0

    def arrival(self, cx: int, cy: int) -> float:
        """Arrival time at a full-map cell, or inf if the sound never gets there."""
        lx, ly = cx - self.x0, cy - self.y0
        if not (0 <= ly < self.t.shape[0] and 0 <= lx < self.t.shape[1]):
            return math.inf
        return float(self.t[ly, lx])

    def arrival_time(self, cx: int, cy: int) -> float:
        """When the sound reaches this cell, in cells of travel."""
        lx, ly = cx - self.x0, cy - self.y0
        if not (0 <= ly < self.travel.shape[0] and 0 <= lx < self.travel.shape[1]):
            return math.inf
        return float(self.travel[ly, lx])

    def remaining(self, cx: int, cy: int) -> float:
        """Energy left on arrival, 1.0 at the source down to 0.0 at the edge."""
        a = self.arrival(cx, cy)
        if not math.isfinite(a):
            return 0.0
        return max(0.0, 1.0 - a / self.energy)

    def bearing(self, cx: int, cy: int) -> tuple[float, float] | None:
        """Perceived direction, as the negative gradient of arrival time.

        Returns a unit (dx, dy) pointing back along the route the sound took,
        or None if the cell was never reached or sits against the window edge.
        """
        lx, ly = cx - self.x0, cy - self.y0
        h, w = self.t.shape
        if not (1 <= ly < h - 1 and 1 <= lx < w - 1):
            return None
        c = float(self.t[ly, lx])
        if not math.isfinite(c):
            return None

        def pick(lo, hi):
            if math.isfinite(lo) and math.isfinite(hi):
                return hi - lo
            if math.isfinite(lo):
                return 2.0 * (c - lo)
            if math.isfinite(hi):
                return 2.0 * (hi - c)
            return 0.0

        gx = float(pick(self.t[ly, lx - 1], self.t[ly, lx + 1]))
        gy = float(pick(self.t[ly - 1, lx], self.t[ly + 1, lx]))
        n = math.hypot(gx, gy)
        if n < 1e-9:
            return None
        return (-gx / n, -gy / n)


class SolveJob:
    """A pure-Python solve that can be spread across frames.

    Dijkstra finalises cells in increasing arrival-time order, so a partially
    completed job is exactly a field that is correct out to some radius and
    unknown beyond it. That is the same shape as the expanding wavefront, so
    the solve can keep pace with the sound instead of blocking a frame: a
    gunshot covering 30k cells costs a couple of milliseconds per frame for a
    few frames rather than 20 ms in one.

    Only the pure-Python backend needs this. skfmm completes fast enough that
    it is done in a single step.
    """

    __slots__ = ("field", "_t", "_tr", "_done", "_heap", "_cost", "_w", "_h",
                 "_energy", "complete", "max_t", "cells")

    def __init__(self, cost_sub, lx, ly, energy, field):
        self.field = field
        self._h, self._w = cost_sub.shape
        self._cost = cost_sub.ravel()
        self._energy = energy
        self._t = [math.inf] * (self._w * self._h)
        self._tr = [math.inf] * (self._w * self._h)
        self._done = bytearray(self._w * self._h)
        start = ly * self._w + lx
        self._t[start] = 0.0
        self._tr[start] = 0.0
        self._heap = [(0.0, start)]
        self.complete = False
        self.max_t = 0.0
        self.cells = 0

    def step(self, budget: int) -> bool:
        """Finalise up to `budget` cells. Returns True when the field is done."""
        if self.complete:
            return True
        w, h = self._w, self._h
        t, tr = self._t, self._tr
        done, heap, flat = self._done, self._heap, self._cost
        INF = math.inf
        push, pop = heapq.heappush, heapq.heappop
        max_t = self.max_t
        touched = []

        n = 0
        while heap and n < budget:
            tv, i = pop(heap)
            if done[i]:
                continue
            if tv > self._energy:
                heap.clear()
                break
            done[i] = 1
            touched.append(i)
            max_t = tv
            n += 1
            x = i % w
            y = i // w
            for j in ((i - 1) if x > 0 else -1,
                      (i + 1) if x < w - 1 else -1,
                      (i - w) if y > 0 else -1,
                      (i + w) if y < h - 1 else -1):
                if j < 0 or done[j]:
                    continue
                f = flat[j]
                if not math.isfinite(f):
                    continue
                nx = j % w
                ny = j // w
                a = INF
                ta = INF
                if nx > 0 and t[j - 1] < a:
                    a, ta = t[j - 1], tr[j - 1]
                if nx < w - 1 and t[j + 1] < a:
                    a, ta = t[j + 1], tr[j + 1]
                b = INF
                tb = INF
                if ny > 0 and t[j - w] < b:
                    b, tb = t[j - w], tr[j - w]
                if ny < h - 1 and t[j + w] < b:
                    b, tb = t[j + w], tr[j + w]

                if a == INF and b == INF:
                    continue
                if a == INF:
                    nt, ntr = b + f, tb + 1.0
                elif b == INF:
                    nt, ntr = a + f, ta + 1.0
                else:
                    d = a - b
                    if -f < d < f:
                        nt = (a + b + math.sqrt(2.0 * f * f - d * d)) * 0.5
                        dt = ta - tb
                        if -1.0 < dt < 1.0:
                            ntr = (ta + tb + math.sqrt(2.0 - dt * dt)) * 0.5
                        else:
                            ntr = min(ta, tb) + 1.0
                    elif a < b:
                        nt, ntr = a + f, ta + 1.0
                    else:
                        nt, ntr = b + f, tb + 1.0

                if nt < t[j] - 1e-6:
                    t[j] = nt
                    tr[j] = ntr
                    push(heap, (nt, j))

        if touched:
            idx = np.asarray(touched, dtype=np.intp)
            self.field.t.reshape(-1)[idx] = np.asarray(
                [t[i] for i in touched], dtype=np.float32)
            self.field.travel.reshape(-1)[idx] = np.asarray(
                [tr[i] for i in touched], dtype=np.float32)
            mt = max(tr[i] for i in touched)
            if mt > self.field.max_travel:
                self.field.max_travel = mt
        self.max_t = max_t
        self.cells += n
        if not heap:
            self.complete = True
        return self.complete


def begin(cost: np.ndarray, origin: tuple[int, int], energy: float,
          backend: str | None = None):
    """Start a solve. Returns (SoundField, job).

    `job` is None when the backend completed synchronously; otherwise call
    job.step(budget) each frame until it returns True. The field is safe to
    read at any point - unreached cells are simply inf.
    """
    backend = backend or BACKEND
    if backend in ("skfmm", "numba"):
        return solve(cost, origin, energy, backend), None

    h, w = cost.shape
    ox, oy = origin
    reach = int(math.ceil(energy / max(float(np.nanmin(cost)), 1e-6))) + 2
    x0 = max(0, ox - reach)
    x1 = min(w, ox + reach + 1)
    y0 = max(0, oy - reach)
    y1 = min(h, oy + reach + 1)
    sub = np.ascontiguousarray(cost[y0:y1, x0:x1], dtype=np.float64)
    lx, ly = ox - x0, oy - y0
    if not math.isfinite(sub[ly, lx]):
        raise ValueError(f"sound origin {origin} sits in an impassable cell")

    field = SoundField(t=np.full(sub.shape, np.inf, dtype=np.float32),
                       travel=np.full(sub.shape, np.inf, dtype=np.float32),
                       x0=x0, y0=y0, origin=origin, energy=energy,
                       backend=backend)
    return field, SolveJob(sub, lx, ly, energy, field)


def solve(cost: np.ndarray, origin: tuple[int, int], energy: float,
          backend: str | None = None) -> SoundField:
    """Solve one sound event.

    `cost` is the per-cell sound cost array (np.inf for truly impassable).
    `origin` is a full-map (cx, cy). `energy` is the loudness budget in cell
    units: the field stops expanding once arrival time exceeds it, so a
    footstep might use 20 and a gunshot 400.

    The solve is cropped to the box the sound could possibly reach, which is
    what keeps quiet sounds cheap regardless of map size.
    """
    backend = backend or BACKEND
    h, w = cost.shape
    ox, oy = origin

    reach = int(math.ceil(energy / max(float(np.nanmin(cost)), 1e-6))) + 2
    x0 = max(0, ox - reach)
    x1 = min(w, ox + reach + 1)
    y0 = max(0, oy - reach)
    y1 = min(h, oy + reach + 1)
    sub = np.ascontiguousarray(cost[y0:y1, x0:x1], dtype=np.float64)

    lx, ly = ox - x0, oy - y0
    if not math.isfinite(sub[ly, lx]):
        raise ValueError(f"sound origin {origin} sits in an impassable cell")

    if backend == "numba":
        if _njit is None:
            raise RuntimeError("numba is not installed")
        t, travel = _solve_numba(sub, lx, ly, float(energy))
        t = t.astype(np.float32)
        travel = travel.astype(np.float32)
        f = SoundField(t=t, travel=travel, x0=x0, y0=y0, origin=origin,
                       energy=energy, backend=backend)
        fin = travel[np.isfinite(travel)]
        f.max_travel = float(fin.max()) if fin.size else 0.0
        return f

    if backend == "skfmm":
        if _skfmm is None:
            raise RuntimeError("scikit-fmm is not installed")
        t = _solve_skfmm(sub, lx, ly, energy)
        uniform = np.where(np.isfinite(sub), 1.0, np.inf)
        travel = _solve_skfmm(uniform, lx, ly, np.inf)
        travel = np.where(np.isfinite(t), travel, np.inf).astype(np.float32)
        f = SoundField(t=t, travel=travel, x0=x0, y0=y0, origin=origin,
                       energy=energy, backend=backend)
        fin = travel[np.isfinite(travel)]
        f.max_travel = float(fin.max()) if fin.size else 0.0
        return f
    if backend != "python":
        raise ValueError(f"unknown backend {backend!r}")

    field = SoundField(t=np.full(sub.shape, np.inf, dtype=np.float32),
                       travel=np.full(sub.shape, np.inf, dtype=np.float32),
                       x0=x0, y0=y0, origin=origin, energy=energy,
                       backend=backend)
    job = SolveJob(sub, lx, ly, energy, field)
    while not job.step(1 << 20):
        pass
    return field
