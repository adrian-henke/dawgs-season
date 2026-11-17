"""Clean up team logos so they match the others on the site.

For each image given (or every .png/.jpg/.jpeg/.webp in the site folder if none are given):
  1. Removes the solid background (usually white) that touches the edges of the image.
     White *inside* the logo (letters, eyes, etc.) is kept, because only background
     connected to the image border is removed. Small gaps in a logo's outline are sealed
     first so the removal can't leak through them.
  2. Softens the edge where the logo meets the removed background, so there's no white halo.
  3. Trims empty space, then centers the logo on a 512 x 512 transparent canvas, scaled so its
     outermost point touches a circle of radius 232. The site shows logos in round badges,
     so every logo (round, wide or tall) fills the badge the same way and stays clear of its border.
  4. Saves it as a .png with the same name (a .jpg becomes a .png and the .jpg is removed).

Logos that are already 512 x 512 with transparent corners are left alone, so it is safe to
run on the whole folder any time. Use --refit to re-fit them anyway (after changing RADIUS).

--fill-holes [N]  For logos that come with the background already cut out, but with the white
                  parts of the logo cut out too (e.g. white letters showing the page through).
                  Fills transparent areas *enclosed* by the logo with white. N (default 0) seals
                  outline gaps up to N pixels first, for areas that are almost but not fully
                  enclosed. Thin slivers are ignored. Always check the result on a dark background.
--refit           Also re-fit logos that are already clean (e.g. after changing RADIUS).

Needs Pillow:  python3 -m venv venv && venv/bin/pip install pillow
Run:           venv/bin/python tools/clean_logo.py              (whole folder)
               venv/bin/python tools/clean_logo.py Team_Name.png
               venv/bin/python tools/clean_logo.py --fill-holes 3 Team_Name.png
               venv/bin/python tools/clean_logo.py --refit                     (re-fit every logo)
"""
import sys
from collections import deque
from pathlib import Path

from PIL import Image, ImageFilter

SIZE = 512          # final canvas size
RADIUS = 232        # logo's outermost point sits on this circle (the badge border starts at about 244)
TOLERANCE = 40      # how far from the background color still counts as background
EDGE_BAND = 2       # pixels next to the background that get softened
GAP = 3             # outline gaps up to about this many pixels are sealed, so white inside the logo isn't removed
SITE = Path(__file__).resolve().parent.parent


def already_clean(im):
    if im.size != (SIZE, SIZE) or im.mode != "RGBA":
        return False
    w, h = im.size
    return all(im.getpixel(p)[3] == 0 for p in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])


def background_color(px, w, h):
    """Most common color along the border."""
    counts = {}
    for x in range(w):
        for y in (0, h - 1):
            c = px[x, y][:3]
            counts[c] = counts.get(c, 0) + 1
    for y in range(h):
        for x in (0, w - 1):
            c = px[x, y][:3]
            counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get)


def fill_holes(im, seal=0):
    """Put white behind transparent areas that the logo encloses (see --fill-holes)."""
    w, h = im.size
    solid = im.getchannel("A").point(lambda v: 255 if v >= 128 else 0)
    wall = solid.filter(ImageFilter.MaxFilter(2 * seal + 1)) if seal else solid
    sp, wp = solid.load(), wall.load()
    outside = bytearray(w * h)
    q = deque([(x, y) for x in range(w) for y in (0, h - 1)] + [(x, y) for y in range(h) for x in (0, w - 1)])
    while q:
        x, y = q.popleft()
        i = y * w + x
        if outside[i] or wp[x, y]:
            continue
        outside[i] = 1
        if x > 0: q.append((x - 1, y))
        if x < w - 1: q.append((x + 1, y))
        if y > 0: q.append((x, y - 1))
        if y < h - 1: q.append((x, y + 1))
    for _ in range(seal + 1):   # take back the edge pixels the sealing covered
        grow = [i for i in range(w * h) if not outside[i] and not sp[i % w, i // w] and (
            (i % w > 0 and outside[i - 1]) or (i % w < w - 1 and outside[i + 1]) or
            (i >= w and outside[i - w]) or (i < w * (h - 1) and outside[i + w]))]
        for i in grow:
            outside[i] = 1
    hole = Image.new("L", (w, h))
    hole.putdata([255 if not sp[i % w, i // w] and not outside[i] else 0 for i in range(w * h)])
    # Ignore thin slivers: keep only hole areas that are at least ~5 px thick somewhere
    core = hole.filter(ImageFilter.MinFilter(5)).load()
    hp = hole.load()
    keep = Image.new("L", (w, h)); kp = keep.load()
    seen = bytearray(w * h)
    for start in range(w * h):
        sx, sy = start % w, start // w
        if seen[start] or not hp[sx, sy]:
            continue
        comp, thick, q = [], False, deque([(sx, sy)])
        seen[start] = 1
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            thick = thick or core[x, y] > 0
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h and not seen[ny * w + nx] and hp[nx, ny]:
                    seen[ny * w + nx] = 1
                    q.append((nx, ny))
        if thick:
            for x, y in comp:
                kp[x, y] = 255
    white = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    white.putalpha(keep.filter(ImageFilter.MaxFilter(3)))   # 1 px under the outline so the edge blends
    white.alpha_composite(im)
    return white


def circle_fit(im):
    """Center (in the cropped image) and radius of the smallest circle around the visible logo."""
    w, h = im.size
    ap = im.getchannel("A").load()
    pts = []
    for y in range(h):   # each row's leftmost and rightmost visible pixel is enough
        xs = [x for x in range(w) if ap[x, y] >= 64]
        if xs:
            pts += [(xs[0], y), (xs[-1] + 1, y)]
    if not pts:
        return w / 2, h / 2, max(w, h) / 2
    pts = pts[::max(1, len(pts) // 1500)]
    best = None
    cx, cy, step = w / 2, h / 2, max(w, h) / 8
    while step > 0.5:    # coarse-to-fine search for the center that needs the smallest circle
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                x0, y0 = cx + dx * step, cy + dy * step
                r = max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for x, y in pts)
                if best is None or r < best[2]:
                    best = (x0, y0, r)
        cx, cy = best[0], best[1]
        step /= 2
    return best


def clean(path, holes=None, refit=False):
    src = Image.open(path)
    if already_clean(src) and holes is None and not refit:
        print(f"skip   {path.name} (already clean)")
        return
    im = src.convert("RGBA")
    if holes is not None:
        im = fill_holes(im, holes)
    w, h = im.size
    px = im.load()
    bg = background_color(px, w, h)
    # If the border is already mostly transparent, only transparent pixels count as background
    # (otherwise a dark logo on a transparent background could lose its dark parts)
    border_px = [px[x, y] for x in range(w) for y in (0, h - 1)] + [px[x, y] for y in range(h) for x in (0, w - 1)]
    transparent = sum(c[3] < 16 for c in border_px) > len(border_px) / 2

    if not transparent:
        # Add a margin of background color all around, so a logo that touches the image edge
        # doesn't trap slivers of background between itself and the edge
        m = GAP + 3
        padded = Image.new("RGBA", (w + 2 * m, h + 2 * m), bg + (255,))
        padded.paste(im, (m, m))
        im, w, h = padded, w + 2 * m, h + 2 * m
        px = im.load()

    def near_bg(c):
        if transparent:
            return c[3] < 16
        return c[3] < 16 or max(abs(c[i] - bg[i]) for i in range(3)) <= TOLERANCE

    # 1. Flood fill from the border through background-colored pixels.
    #    The logo ("wall") is first thickened by GAP pixels so the fill can't slip through small
    #    gaps in an outline; afterwards the fill is grown back GAP pixels to reach the logo's edge.
    bgish = bytearray(1 if near_bg(px[x, y]) else 0 for y in range(h) for x in range(w))
    wall = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            if not bgish[y * w + x]:
                for yy in range(max(0, y - GAP), min(h, y + GAP + 1)):
                    row = yy * w
                    for xx in range(max(0, x - GAP), min(w, x + GAP + 1)):
                        wall[row + xx] = 1

    def fill(start, ok):
        mask = bytearray(w * h)
        q = deque(start)
        while q:
            x, y = q.popleft()
            i = y * w + x
            if mask[i] or not ok(i):
                continue
            mask[i] = 1
            if x > 0: q.append((x - 1, y))
            if x < w - 1: q.append((x + 1, y))
            if y > 0: q.append((x, y - 1))
            if y < h - 1: q.append((x, y + 1))
        return mask

    border = [(x, y) for x in range(w) for y in (0, h - 1)] + [(x, y) for y in range(h) for x in (0, w - 1)]
    is_bg = fill(border, lambda i: not wall[i])
    for _ in range(GAP):   # grow back into background-colored pixels only
        grow = [i for i in range(w * h) if not is_bg[i] and bgish[i] and (
            (i % w > 0 and is_bg[i - 1]) or (i % w < w - 1 and is_bg[i + 1]) or
            (i >= w and is_bg[i - w]) or (i < w * (h - 1) and is_bg[i + w]))]
        for i in grow:
            is_bg[i] = 1

    # 2. Distance (in pixels, up to EDGE_BAND) from the background, for edge softening
    dist = [0 if b else EDGE_BAND + 1 for b in is_bg]
    for step in range(1, EDGE_BAND + 1):
        for y in range(h):
            for x in range(w):
                i = y * w + x
                if dist[i] > step and any(
                    0 <= nx < w and 0 <= ny < h and dist[ny * w + nx] == step - 1
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                ):
                    dist[i] = step

    for y in range(h):
        for x in range(w):
            i = y * w + x
            if is_bg[i]:
                px[x, y] = (0, 0, 0, 0)
            elif dist[i] <= EDGE_BAND and not transparent:
                # Pixel is a blend of the logo color and the background: work out how much is logo
                r, g, b, a0 = px[x, y]
                a = max(abs(r - bg[0]), abs(g - bg[1]), abs(b - bg[2])) / max(1, max(bg[0], bg[1], bg[2], 255 - min(bg)))
                a = min(1.0, max(0.0, a))
                if a < 1.0 and a > 0.0:
                    un = lambda c, k: max(0, min(255, round((c - (1 - a) * bg[k]) / a)))
                    px[x, y] = (un(r, 0), un(g, 1), un(b, 2), round(a * a0))
                elif a == 0.0:
                    px[x, y] = (0, 0, 0, 0)

    # 3. Trim, scale so the logo's outermost point touches the RADIUS circle, center on the canvas
    box = im.getchannel("A").getbbox()
    if not box:
        print(f"error  {path.name}: nothing left after removing the background")
        return
    im = im.crop(box)
    cx, cy, r = circle_fit(im)
    scale = RADIUS / r
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    out = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    out.alpha_composite(im, (round(SIZE / 2 - cx * scale), round(SIZE / 2 - cy * scale)))

    # 4. Save as .png
    dest = path.with_suffix(".png")
    out.save(dest, optimize=True)
    if dest != path:
        path.unlink()
    print(f"clean  {path.name} -> {dest.name} (" + ("already transparent; trimmed and resized" if transparent else f"background {bg} removed") + ")")


if __name__ == "__main__":
    args, holes, refit = sys.argv[1:], None, False
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--fill-holes":
            holes = int(args.pop(0)) if args and args[0].isdigit() else 0
        elif flag == "--refit":
            refit = True
    files = [Path(a).resolve() for a in args] or sorted(
        p for p in SITE.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    for f in files:
        clean(f, holes, refit)
