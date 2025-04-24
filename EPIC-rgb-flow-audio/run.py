#python train_video_flow_audio_EPIC_SimMMDG.py  --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --datapath D:\EPIC_KITCHENS
#python train_video_flow_audio_EPIC_SimMMDG.py  --use_video --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --datapath D:\EPIC_KITCHENS
#python train_video_flow_audio_EPIC_SimMMDG.py  --use_video --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --datapath D:\EPIC_KITCHENS

import os

cmd1= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --datapath D:\EPIC_KITCHENS --appen r1"
# cmd2= "python train_video_flow_audio_EPIC_SimMMDG.py  --use_video --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --datapath D:\EPIC_KITCHENS --appen r1"
# cmd3= "python train_video_flow_audio_EPIC_SimMMDG.py  --use_video --use_flow -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --datapath D:\EPIC_KITCHENS --appen r1"
os.system(cmd1)
# os.system(cmd2)
# os.system(cmd3)
