"""Generate a clean portfolio/cap-table fixture image for the Card -> Rows demo."""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 520
img = Image.new("RGB", (W, H), "#ffffff")
d = ImageDraw.Draw(img)


def font(size, bold=False):
    for p in ([r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"] if bold
              else [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Header band
d.rectangle([0, 0, W, 70], fill="#0b1220")
d.text((34, 22), "Evergreen Ventures — Portfolio (FY2024)", font=font(28, True), fill="#eaf2ff")

cols = ["Company", "Round", "Raised", "Valuation", "Lead Investor"]
xs = [34, 300, 440, 590, 760]
d.text((34, 92), "", font=font(18), fill="#0f172a")
for x, c in zip(xs, cols):
    d.text((x, 96), c, font=font(20, True), fill="#0f172a")
d.line([24, 128, W - 24, 128], fill="#cbd5e1", width=2)

rows = [
    ("Acme Robotics", "Series B", "$40M", "$300M", "Sequoia"),
    ("Globex Logistics", "Series A", "$18M", "$110M", "General Catalyst"),
    ("Initech AI", "Seed", "$6M", "$35M", "a16z"),
    ("Umbrella Health", "Series C", "$120M", "$1.2B", "Thrive Capital"),
    ("Hooli Cloud", "Series B", "$55M", "$420M", "Index Ventures"),
]
y = 146
for r in rows:
    for x, cell in zip(xs, r):
        d.text((x, y), cell, font=font(19, x == 34), fill="#0f172a")
    d.line([24, y + 34, W - 24, y + 34], fill="#eef2f7", width=1)
    y += 56

d.text((34, y + 12), "Confidential — internal portfolio review", font=font(15), fill="#64748b")

out = os.path.join(os.path.dirname(__file__), "cap_table.png")
img.save(out)
print("wrote", out)
