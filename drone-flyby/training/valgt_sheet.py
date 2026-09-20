"""Contact sheet of track crops for visual verification."""
import json, sys, cv2, numpy as np
OUT='/mnt/data/nordicai/drone-flyby/valgt/'
ids=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 and sys.argv[2] != 'all' else None
tracks={t['id']:t for t in json.load(open(OUT+'tracks.json'))}
sel=[tracks[i] for i in ids] if ids else list(tracks.values())
tiles=[]
for t in sel:
    d=t['best_det']; img=cv2.imread(d['png']); x1,y1,x2,y2=d['region']; s=(x2-x1)/960
    b=[(d['box'][0]-x1)/s,(d['box'][1]-y1)/s,(d['box'][2]-x1)/s,(d['box'][3]-y1)/s]
    cx,cy=int((b[0]+b[2])/2),int((b[1]+b[3])/2); half=max(28,int(max(b[2]-b[0],b[3]-b[1])*0.9))
    pad=cv2.copyMakeBorder(img,200,200,200,200,cv2.BORDER_CONSTANT)
    crop=pad[cy+200-half:cy+200+half, cx+200-half:cx+200+half]
    k=180/crop.shape[0]; crop=cv2.resize(crop,(180,180),interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop,(int((b[0]-cx+half)*k),int((b[1]-cy+half)*k)),(int((b[2]-cx+half)*k),int((b[3]-cy+half)*k)),(0,0,255),1)
    tile=np.zeros((210,180,3),np.uint8); tile[30:]=crop
    cv2.putText(tile,f"{t['id']} {t['class'][:12]}",(2,12),0,0.38,(0,255,255),1)
    cv2.putText(tile,f"c{t['max_conf']:.2f} f{t['n_frames']} {''.join(m[0] for m in t['models_conf30'])}",(2,26),0,0.36,(255,255,255),1)
    tiles.append(tile)
cols=8
while len(tiles)%cols: tiles.append(np.zeros((210,180,3),np.uint8))
sheet=np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)])
cv2.imwrite(sys.argv[1],sheet,[cv2.IMWRITE_JPEG_QUALITY,90]); print(len(sel),'tiles ->',sys.argv[1])
