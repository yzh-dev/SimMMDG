import os
path = "D:\EPIC_KITCHENS"
# Video and Flow and Audio
# D1 best:63.21%
cmd1= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 10 --datapath {} --appen _vfa_t_D1_epoch10_gradeclip10".format(path)
# D2 best:68.4%
# cmd2= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 10 --datapath {} --appen _vfa_t_D1_epoch15_alphatrans0.5".format(path)
# # D3 best:67.45%
# cmd3= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --alpha_trans 0.01 --datapath {} --appen _vfa_t_D1_epoch15_alphatrans0.01".format(path)


# cmd2= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 10 --datapath {}  --appen _vfa_t_D2_epoch10".format(path)
# cmd3= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 15 --alpha_trans 1.0 --datapath {}  --appen _vfa_t_D3_epoch15_alphatrans1.0".format(path)

# Video and Audio
# cmd4= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath {}  --appen _va_t_D1_epoch20".format(path)
# cmd5= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath {}  --appen _va_t_D2_epoch20".format(path)
# cmd6= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 25 --datapath {}  --appen _va_t_D3_epoch25".format(path)

# # Video and Flow
# cmd7= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --datapath {}  --appen _vf_t_D1_epoch15".format(path)
# cmd8= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 15 --datapath {}  --appen _vf_t_D2_epoch15".format(path)
# cmd9= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 15 --datapath {}  --appen _vf_t_D3_epoch15".format(path)

# # Flow and Audio
# cmd10= "python train_video_flow_audio_EPIC_SimMMDG.py --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath {}  --appen _fa_t_D1_epoch20".format(path)
# cmd11= "python train_video_flow_audio_EPIC_SimMMDG.py --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath {}  --appen _fa_t_D2_epoch20".format(path)
# cmd12= "python train_video_flow_audio_EPIC_SimMMDG.py --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath {}  --appen _fa_t_D3_epoch20".format(path)

os.system(cmd1)
# os.system(cmd2)
# os.system(cmd3)
# os.system(cmd4)
# os.system(cmd5)
# os.system(cmd6)
# os.system(cmd7)
# os.system(cmd8)
# os.system(cmd9)
# os.system(cmd10)    
# os.system(cmd11)
# os.system(cmd12)

# 下步调试，都设置为25轮，观察下不同域的最好结果

