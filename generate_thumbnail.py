"""
Generate a YouTube thumbnail for claudecodex.
Requires: pip install pillow
Run: python generate_thumbnail.py
Output: thumbnail.png (1280x720)
"""

from PIL import Image, ImageDraw, ImageFont
import sys
import os

W, H = 1280, 720

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#0D1117"   # GitHub dark
CARD        = "#161B22"   # slightly lighter card
BORDER      = "#30363D"   # subtle border
ACCENT      = "#7C3AED"   # purple
ACCENT2     = "#06B6D4"   # cyan
WHITE       = "#F0F6FC"
MUTED       = "#8B949E"
ARROW       = "#F59E0B"   # amber

def hex_color(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def load_font(size, bold=False):
    """Try to load a system font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def draw_rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=width)

def center_text(draw, text, font, y, color, x_start=0, x_end=W):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = x_start + (x_end - x_start - tw) // 2
    draw.text((x, y), text, font=font, fill=color)

def draw_node(draw, cx, cy, label, sublabel, font_main, font_sub, color):
    bw, bh = 200, 72
    x0, y0 = cx - bw//2, cy - bh//2
    draw_rounded_rect(draw, [x0, y0, x0+bw, y0+bh], radius=10,
                      fill=hex_color(CARD), outline=hex_color(color), width=2)
    # label
    bbox = draw.textbbox((0, 0), label, font=font_main)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw//2, y0 + 10), label, font=font_main, fill=hex_color(WHITE))
    # sublabel
    bbox2 = draw.textbbox((0, 0), sublabel, font=font_sub)
    tw2 = bbox2[2] - bbox2[0]
    draw.text((cx - tw2//2, y0 + 38), sublabel, font=font_sub, fill=hex_color(color))

def draw_arrow(draw, x0, y0, x1, y1, color, label="", font=None):
    """Draw horizontal arrow with optional label above."""
    mid_x = (x0 + x1) // 2
    mid_y = (y0 + y1) // 2
    draw.line([(x0, mid_y), (x1, mid_y)], fill=hex_color(color), width=3)
    # arrowhead
    ah = 10
    draw.polygon([(x1, mid_y), (x1-ah, mid_y-ah//2), (x1-ah, mid_y+ah//2)],
                 fill=hex_color(color))
    if label and font:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((mid_x - tw//2, mid_y - 20), label, font=font, fill=hex_color(MUTED))

# ── Build image ───────────────────────────────────────────────────────────────
img = Image.new("RGB", (W, H), hex_color(BG))
draw = ImageDraw.Draw(img)

# Fonts
f_title   = load_font(82, bold=True)
f_sub     = load_font(34)
f_node    = load_font(22, bold=True)
f_nodesub = load_font(16)
f_arrow   = load_font(15)
f_badge   = load_font(20)
f_tag     = load_font(18)

# ── Subtle grid background ─────────────────────────────────────────────────
for x in range(0, W, 60):
    draw.line([(x, 0), (x, H)], fill=(255, 255, 255, 8), width=1)
for y in range(0, H, 60):
    draw.line([(0, y), (W, y)], fill=(255, 255, 255, 8), width=1)

# ── Glow blobs (simulate) ────────────────────────────────────────────────────
for r in range(180, 0, -6):
    alpha = max(1, int(20 * (1 - r / 180)))
    draw.ellipse([120-r, 80-r, 120+r, 80+r],
                 fill=(124, 58, 237, alpha))   # purple glow top-left

for r in range(140, 0, -5):
    alpha = max(1, int(18 * (1 - r / 140)))
    draw.ellipse([W-140-r, H-80-r, W-140+r, H-80+r],
                 fill=(6, 182, 212, alpha))    # cyan glow bottom-right

# ── Title ─────────────────────────────────────────────────────────────────────
title = "claudecodex"
bbox = draw.textbbox((0, 0), title, font=f_title)
tw = bbox[2] - bbox[0]
tx = (W - tw) // 2
ty = 58

# Purple shadow
draw.text((tx + 3, ty + 3), title, font=f_title, fill=(124, 58, 237, 120))
# Main white text
draw.text((tx, ty), title, font=f_title, fill=hex_color(WHITE))

# Accent underline
ul_y = ty + (bbox[3] - bbox[1]) + 6
draw.rounded_rectangle([tx, ul_y, tx + tw, ul_y + 5], radius=3, fill=hex_color(ACCENT))

# ── Subtitle ──────────────────────────────────────────────────────────────────
subtitle = "Escape rate limits  ·  Plug in any LLM backend"
center_text(draw, subtitle, f_sub, ul_y + 20, MUTED)

# ── Flow diagram ──────────────────────────────────────────────────────────────
diag_y = 330
left_x   = 140
proxy_x  = W // 2
right_x  = W - 160

# Claude Code node
draw_node(draw, left_x + 60, diag_y, "Claude Code", "VS Code / Terminal",
          f_node, f_nodesub, ACCENT2)

# Arrow: Claude Code → proxy
draw_arrow(draw, left_x + 165, diag_y, proxy_x - 115, diag_y,
           ARROW, "HTTP requests", f_arrow)

# Proxy node (larger, centered)
px0, py0 = proxy_x - 110, diag_y - 50
px1, py1 = proxy_x + 110, diag_y + 50
draw_rounded_rect(draw, [px0, py0, px1, py1], radius=12,
                  fill=hex_color(CARD), outline=hex_color(ACCENT), width=3)
center_text(draw, "claudecodex", f_node, py0 + 10, WHITE, px0, px1)
center_text(draw, "proxy + translator", f_nodesub, py0 + 36, ACCENT, px0, px1)

# Arrow: proxy → providers
draw_arrow(draw, proxy_x + 115, diag_y, right_x - 65, diag_y,
           ARROW, "translated API", f_arrow)

# Providers stacked on right
provider_cx = right_x - 60
providers = [
    ("Bedrock",  "#F59E0B"),
    ("Gemini",   "#06B6D4"),
    ("Ollama",   "#4ADE80"),
]
spacing = 56
start_y = diag_y - spacing
for i, (name, color) in enumerate(providers):
    py = start_y + i * spacing
    bw, bh = 140, 42
    bx = provider_cx - bw // 2
    draw_rounded_rect(draw, [bx, py - bh//2, bx + bw, py + bh//2],
                      radius=8, fill=hex_color(CARD),
                      outline=hex_color(color), width=2)
    bbox = draw.textbbox((0, 0), name, font=f_badge)
    tw2 = bbox[2] - bbox[0]
    draw.text((provider_cx - tw2//2, py - (bbox[3]-bbox[1])//2),
              name, font=f_badge, fill=hex_color(color))

# ── Bottom badges ─────────────────────────────────────────────────────────────
tags = ["Open Source", "No Rate Limits", "AWS Bedrock", "Gemini", "Local LLMs"]
tag_y = H - 72
gap = 16
total_w = 0
widths = []
for t in tags:
    bbox = draw.textbbox((0, 0), t, font=f_tag)
    w = bbox[2] - bbox[0] + 28
    widths.append(w)
    total_w += w + gap
total_w -= gap
start_x = (W - total_w) // 2

colors_cycle = [ACCENT, ACCENT2, "#F59E0B", "#4ADE80", "#F472B6"]
x_cur = start_x
for i, (t, tw3) in enumerate(zip(tags, widths)):
    c = colors_cycle[i % len(colors_cycle)]
    draw_rounded_rect(draw, [x_cur, tag_y, x_cur + tw3, tag_y + 34],
                      radius=17, fill=(0, 0, 0, 0), outline=hex_color(c), width=2)
    bbox = draw.textbbox((0, 0), t, font=f_tag)
    th = bbox[3] - bbox[1]
    draw.text((x_cur + 14, tag_y + (34 - th)//2), t, font=f_tag, fill=hex_color(c))
    x_cur += tw3 + gap

# ── Save ───────────────────────────────────────────────────────────────────────
out = "thumbnail.png"
img.save(out, "PNG")
print(f"Saved → {out}  ({W}×{H})")
