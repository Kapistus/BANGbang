"""Headless test of the fog window.

The veil used to be built and blurred across the whole map every frame, which
on the 80x52 m vessel map was the single biggest cost in the frame. It is now
built only for the fine cells under the camera, plus a margin wide enough that
the blur has something to read at the edges.

That is only safe if the window gives exactly the same veil as the whole map
would, so this compares the two, pixel for pixel, on the plain fog and on the
ripple view - which is the one that also carries colour.

What this checks:

  * every cell that reaches the screen matches a full-map build, for a camera
    in a corner, in the middle, and hard against the far edge
  * the window really is a window: small next to the map, and never outside it
  * the blur margin is wide enough that no cell on screen is blurred against
    cells the window left out

Run from the project root:   python -m tests.fog_window_test
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

import main as renderer

H, W = 416, 640          # the vessel map's fine grid
VIEW_W, VIEW_H = 1860, 776
PPC = 6.0                # pixels per fine cell at the game's zoom


def _world(seed=7):
    rng = np.random.default_rng(seed)
    known = rng.random((H, W)) > 0.35
    # a vision field somewhere in the middle of the map, as the game has it
    fy0, fx0, fh, fw = 120, 180, 96, 128
    inten = np.clip(rng.random((fh, fw)) * 1.4 - 0.3, 0.0, 1.0).astype(np.float32)
    mflash = np.zeros((fh, fw), np.float32)
    mflash[40:60, 50:70] = rng.random((20, 20)).astype(np.float32)
    own = np.zeros((H, W), np.float32)
    own[130:190, 200:260] = rng.random((60, 60)).astype(np.float32) * 255.0
    enemy = np.zeros((H, W), np.float32)
    enemy[150:210, 240:300] = rng.random((60, 60)).astype(np.float32) * 255.0
    return known, (fy0, fx0, inten), mflash, (own, enemy)


def _full(known, cone, mflash, ripple):
    """The old way: the whole map, every frame."""
    return renderer.fog_veil(known, (0, H, 0, W), cone=cone, mflash=mflash,
                             ripple=ripple)


def compare_test():
    known, cone, mflash, ripple = _world()
    cams = {"corner": (0, 0),
            "middle": (int(W * PPC / 2) - VIEW_W // 2,
                       int(H * PPC / 2) - VIEW_H // 2),
            "far edge": (int(W * PPC) - VIEW_W, int(H * PPC) - VIEW_H)}
    for name, (cam_x, cam_y) in cams.items():
        cam_x, cam_y = max(0, cam_x), max(0, cam_y)
        build, blit = renderer.fog_windows((H, W), cam_x, cam_y, VIEW_W,
                                           VIEW_H, PPC)
        by0, by1, bx0, bx1 = build
        sy0, sy1, sx0, sx1 = blit
        cells = (by1 - by0) * (bx1 - bx0)
        assert 0 <= by0 <= sy0 < sy1 <= by1 <= H, (build, blit)
        assert 0 <= bx0 <= sx0 < sx1 <= bx1 <= W, (build, blit)
        for rip in (None, ripple):
            win_a, win_rgb = renderer.fog_veil(known, build, cone=cone,
                                               mflash=mflash, ripple=rip)
            all_a, all_rgb = _full(known, cone, mflash, rip)
            got = win_a[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0]
            want = all_a[sy0:sy1, sx0:sx1]
            err = float(np.abs(got - want).max())
            assert err < 1e-6, \
                f"{name}: the window's veil differs from the whole map's by {err}"
            if rip is not None:
                gotc = win_rgb[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0]
                wantc = all_rgb[sy0:sy1, sx0:sx1]
                cerr = float(np.abs(gotc - wantc).max())
                assert cerr < 1e-3, f"{name}: ripple colour differs by {cerr}"
        print(f"  {name:8}: window {bx1 - bx0}x{by1 - by0} cells "
              f"({cells / (H * W):.0%} of the map), veil identical")
    print("\nFOG WINDOW CHECKS PASSED")


def margin_test():
    """Build on exactly what is drawn and the edge cells blur against nothing.
    The margin is what stops that showing as a seam down the screen."""
    known, cone, mflash, _rip = _world()
    cam = (1000, 400)
    build, blit = renderer.fog_windows((H, W), cam[0], cam[1], VIEW_W, VIEW_H,
                                       PPC)
    by0, by1, bx0, bx1 = build
    sy0, sy1, sx0, sx1 = blit
    assert by0 < sy0 and bx0 < sx0 and by1 > sy1 and bx1 > sx1, \
        "the build window has no margin around what is drawn"
    tight, _ = renderer.fog_veil(known, blit, cone=cone, mflash=mflash)
    full, _ = renderer.fog_veil(known, (0, H, 0, W), cone=cone, mflash=mflash)
    edge = float(np.abs(tight[0] - full[sy0, sx0:sx1]).max())
    win, _ = renderer.fog_veil(known, build, cone=cone, mflash=mflash)
    ok_edge = float(np.abs(win[sy0 - by0, sx0 - bx0:sx1 - bx0]
                           - full[sy0, sx0:sx1]).max())
    print(f"  top row of the drawn area: without the margin off by {edge:.3f}, "
          f"with it off by {ok_edge:.6f}")
    assert edge > 1e-3, "the margin is not doing anything - check FOG_BLUR_R"
    assert ok_edge < 1e-6
    print("\nMARGIN CHECKS PASSED")


if __name__ == "__main__":
    compare_test()
    print()
    margin_test()
    print("\nALL FOG WINDOW TESTS PASSED")
