import os
# path = "D:\EPIC_KITCHENS"
path="D:\ML\Dataset\EPIC_KITCHENS"
# Video and Flow and Audio
# D1 best:63.21%
# cmd1= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 10 --datapath {} --appen _vfa_t_D1_epoch10_gradeclip10".format(path)
# D2 best:68.4%
# cmd2= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 10 --datapath {} --appen _vfa_t_D1_epoch15_alphatrans0.5".format(path)
# # D3 best:67.45%
# cmd3= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 15 --alpha_trans 0.01 --datapath {} --appen _vfa_t_D1_epoch15_alphatrans0.01".format(path)

cmd1= "python train_video_flow_audio_EPIC_SimMMDG.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --nepochs 10 --datapath {} --appen _vfa_t_D1_SimMMDG".format(path)
cmd2= "python train_video_flow_audio_EPIC_SimMMDG_copy.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --nepochs 10 --datapath {} --appen _vfa_t_D1_mutualinfoloss0.5".format(path)

os.system(cmd1)
# os.system(cmd2)



