import cv2
import numpy as np
import os

paths=[
r"C:\Users\gtcam\OneDrive\Desktop\document_scanner\OIP.webp",
r"C:\Users\gtcam\OneDrive\Desktop\document_scanner\3rd image.webp",
r"C:\Users\gtcam\OneDrive\Desktop\document_scanner\2nd image.webp"
]

os.makedirs("outputs",exist_ok=True)

def to3(x):
    return cv2.cvtColor(x,cv2.COLOR_GRAY2BGR)

run=1

for path in paths:

    img=cv2.imread(path)
    if img is None:
        print("Error loading",path)
        continue

    img=cv2.resize(img,(512,512))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    high=gray
    med=cv2.resize(gray,(256,256))
    low=cv2.resize(gray,(128,128))

    med_up=cv2.resize(med,(512,512),interpolation=cv2.INTER_NEAREST)
    low_up=cv2.resize(low,(512,512),interpolation=cv2.INTER_NEAREST)

    q256=gray
    q16=(gray//16)*16
    q4=(gray//64)*64

    cv2.imwrite(f"outputs/run{run}_gray.png",gray)
    cv2.imwrite(f"outputs/run{run}_high.png",high)
    cv2.imwrite(f"outputs/run{run}_med.png",med_up)
    cv2.imwrite(f"outputs/run{run}_low.png",low_up)
    cv2.imwrite(f"outputs/run{run}_q256.png",q256)
    cv2.imwrite(f"outputs/run{run}_q16.png",q16)
    cv2.imwrite(f"outputs/run{run}_q4.png",q4)

    top=np.hstack((img,to3(high),to3(med_up),to3(low_up)))
    cv2.putText(top,"Original",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(top,"512",(560,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(top,"256",(1080,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(top,"128",(1600,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    bottom=np.hstack((to3(q256),to3(q16),to3(q4)))
    # pad bottom row to match top width (4 panels wide) to avoid stacking errors
    pad = np.zeros_like(to3(q256))
    bottom = np.hstack((bottom, pad))
    cv2.putText(bottom,"256 Levels",(60,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(bottom,"16 Levels",(560,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(bottom,"4 Levels",(1060,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    panel=np.vstack((top,bottom))

    cv2.putText(panel,"Loss of fine text details at low resolution",(100,1020),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(panel,"Readability decreases due to blur and banding",(100,1060),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(panel,"OCR best at 512 resolution and 256 gray levels",(100,1100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.imwrite(f"outputs/run{run}_comparison.png",panel)

    cv2.imshow(f"Run {run}",panel)

    run+=1

cv2.waitKey(0)
cv2.destroyAllWindows()