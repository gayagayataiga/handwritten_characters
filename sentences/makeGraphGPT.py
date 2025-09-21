# Render the newly uploaded JSON into a PNG using the same pipeline as before,
# so the user can compare "original" vs "processed" display.
from PIL import Image, ImageDraw
import json, math

INPUT_PATH2 = "sentences/AIと一緒に学習しています。_20250920145922.json"
OUT_PATH2 = "sentences/sample_render_original.png"

with open(INPUT_PATH2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

seq2 = data2.get("sequence", [])
coords2 = []
x,y = 0.0,0.0
for dx,dy,pen in seq2:
    x += dx; y += dy
    coords2.append((x,y,pen))

strokes2 = []
cur=[]
for (x,y,pen) in coords2:
    cur.append((x,y))
    if pen==1:
        strokes2.append(cur)
        cur=[]
if cur: strokes2.append(cur)

all_xy2 = [(px,py) for s in strokes2 for (px,py) in s] if strokes2 else [(0,0)]
xs2 = [p[0] for p in all_xy2]; ys2 = [p[1] for p in all_xy2]
minx2,maxx2 = min(xs2),max(xs2)
miny2,maxy2 = min(ys2),max(ys2)
w2 = maxx2-minx2 if maxx2>minx2 else 1.0
h2 = maxy2-miny2 if maxy2>miny2 else 1.0

canvas_w,canvas_h = 1600,400
pad=60
scale2 = min((canvas_w-2*pad)/w2, (canvas_h-2*pad)/h2)
tx2 = -minx2+(canvas_w/scale2 - w2)/2
ty2 = -miny2+(canvas_h/scale2 - h2)/2

def to_canvas2(pt):
    x,y = pt
    return ((x+tx2)*scale2, (y+ty2)*scale2)

img2 = Image.new("RGB", (canvas_w,canvas_h), (255,255,255))
draw2 = ImageDraw.Draw(img2)

for s in strokes2:
    if len(s)<2: continue
    speeds2=[]
    for i in range(1,len(s)):
        x0,y0=s[i-1]; x1,y1=s[i]
        dist=math.hypot(x1-x0,y1-y0); speeds2.append(dist)
    if speeds2: smin,smax=min(speeds2),max(speeds2)
    else: smin,smax=0.0,1.0
    lw_base=2.0
    pts2=[to_canvas2(p) for p in s]
    for i in range(1,len(pts2)):
        p0,p1=pts2[i-1],pts2[i]
        speed=speeds2[i-1] if speeds2 else 0.5
        if smax-smin>1e-6:
            norm=(speed-smin)/(smax-smin)
        else: norm=0.5
        width=lw_base+(1.5-norm*1.2)
        draw2.line([p0,p1],fill=(10,10,10),width=int(max(1,round(width))))

for yline in range(80,canvas_h,80):
    draw2.line([(0,yline),(canvas_w,yline)],fill=(240,240,255),width=1)

img2.save(OUT_PATH2)
from IPython.display import Image as IPyImage, display
display(IPyImage(OUT_PATH2))
print(f"Saved second rendering to: {OUT_PATH2}")
