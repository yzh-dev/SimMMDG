from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments
from dataloader_video_flow_audio_EPIC_SimMMDG import EPICDOMAIN
import torch.nn.functional as F
from losses import SupConLoss
import wandb
wandb.login(key="")
# 需要进一步debug观察形状变化
def train_one_step(clip, labels, flow, spectrogram):
    labels = labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)#shape:[batch_size, 3, 32, 224, 224]
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1) #shape:[batch_size, 2, 32, 224, 224]
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).cuda() #shape:[batch_size, 1, 257, 1004]

    with torch.no_grad():
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)   # shape:[batch_size,1024,8,14,14]
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)  # x_slow:[batch_size,1280,8,14,14] x_fast:[batch_size,128,32,14,14]
            v_feat = (x_slow.detach(), x_fast.detach())  
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)  # 对音频数据，首先进行通用特征的提取audio_feat:[batch_size,256,17,63]

    if args.use_video:
        v_feat = model.module.backbone.get_predict(v_feat)  # 使用layer4进一步提取数据，分别为[batch_size,2048,8,7,7]和[batch_size,256,32,7,7]
        predict1, v_emd = model.module.cls_head(v_feat)  # predict1形状为[batch_size, 8]，v_emd为视频模态专用特征+模态共享特征，v_emd形状为[batch_size, 2304]
        v_dim = int(v_emd.shape[1] / 2)  # 视频模态专有特征维度

    if args.use_flow:
        f_feat = model_flow.module.backbone.get_predict(f_feat.detach())# [batch_size,2048,8,7,7]
        f_predict, f_emd = model_flow.module.cls_head(f_feat) # f_predict形状为[batch_size, 8]，f_emd为光流模态专用特征+模态共享特征，f_emd形状为[batch_size, 8]，f_emd形状为[batch_size, 2048]
        f_dim = int(f_emd.shape[1] / 2)

    if args.use_audio:    
        audio_predict, audio_emd = audio_cls_model(audio_feat.detach())  # audio_predict形状为[batch_size, 8]，audio_emd为音频模态专用特征+模态共享特征，audio_emd形状为[batch_size, 512]
        a_dim = int(audio_emd.shape[1] / 2)

    if args.use_video and args.use_flow and args.use_audio:
        feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)  # 输入包括所有模态的特征，输入形状为[batch_size, 2304+2048+512]
    elif args.use_video and args.use_flow:
        feat = torch.cat((v_emd, f_emd), dim=1)
    elif args.use_video and args.use_audio:
        feat = torch.cat((v_emd, audio_emd), dim=1)
    elif args.use_flow and args.use_audio:
        feat = torch.cat((f_emd, audio_emd), dim=1)

    predict = mlp_cls(feat)  # 输入包括所有模态的特征，输出为8个类别的预测概率，输入形状为[batch_size, 2304+2048+512]，输出形状为[batch_size, 8]
    # 当 train_one_step 函数在 main 函数内被调用时，它形成了一个闭包。闭包可以访问其定义环境中的变量，包括外部函数的局部变量
    loss = criterion(predict, labels)  # 分类损失

    # Cross-modal Translation 首先计算跨模态转换损失，用到了模态专用特征+模态共享特征
    if args.use_video and args.use_flow and args.use_audio:
        a_emd_t = mlp_v2a(v_emd)  
        v_emd_t = mlp_a2v(audio_emd)  
        f_emd_t = mlp_v2f(v_emd)   # 视频模态转换为光流模态，形状从[batch_size, 2304]转换为[batch_size, 2048]
        v_emd_t2 = mlp_f2v(f_emd)
        a_emd_t2 = mlp_f2a(f_emd)
        f_emd_t2 = mlp_a2f(audio_emd)
        a_emd_t = a_emd_t/torch.norm(a_emd_t, dim=1, keepdim=True)
        v_emd_t = v_emd_t/torch.norm(v_emd_t, dim=1, keepdim=True)
        f_emd_t = f_emd_t/torch.norm(f_emd_t, dim=1, keepdim=True)
        a_emd_t2 = a_emd_t2/torch.norm(a_emd_t2, dim=1, keepdim=True)
        v_emd_t2 = v_emd_t2/torch.norm(v_emd_t2, dim=1, keepdim=True)
        f_emd_t2 = f_emd_t2/torch.norm(f_emd_t2, dim=1, keepdim=True)
        v2a_loss = torch.mean(torch.norm(a_emd_t-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2v_loss = torch.mean(torch.norm(v_emd_t-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        v2f_loss = torch.mean(torch.norm(f_emd_t-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        f2a_loss = torch.mean(torch.norm(a_emd_t2-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        f2v_loss = torch.mean(torch.norm(v_emd_t2-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        a2f_loss = torch.mean(torch.norm(f_emd_t2-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2a_loss + a2v_loss+ v2f_loss+ f2a_loss+ f2v_loss+ a2f_loss)/6
    elif args.use_video and args.use_flow:
        f_emd_t = mlp_v2f(v_emd)
        v_emd_t2 = mlp_f2v(f_emd)
        f_emd_t = f_emd_t/torch.norm(f_emd_t, dim=1, keepdim=True)
        v_emd_t2 = v_emd_t2/torch.norm(v_emd_t2, dim=1, keepdim=True)
        v2f_loss = torch.mean(torch.norm(f_emd_t-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        f2v_loss = torch.mean(torch.norm(v_emd_t2-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2f_loss+ f2v_loss)/2
    elif args.use_video and args.use_audio:
        a_emd_t = mlp_v2a(v_emd)
        v_emd_t = mlp_a2v(audio_emd)
        a_emd_t = a_emd_t/torch.norm(a_emd_t, dim=1, keepdim=True)
        v_emd_t = v_emd_t/torch.norm(v_emd_t, dim=1, keepdim=True)
        v2a_loss = torch.mean(torch.norm(a_emd_t-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2v_loss = torch.mean(torch.norm(v_emd_t-v_emd/torch.norm(v_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(v2a_loss + a2v_loss)/2
    elif args.use_flow and args.use_audio:
        a_emd_t2 = mlp_f2a(f_emd)
        f_emd_t2 = mlp_a2f(audio_emd)
        a_emd_t2 = a_emd_t2/torch.norm(a_emd_t2, dim=1, keepdim=True)
        f_emd_t2 = f_emd_t2/torch.norm(f_emd_t2, dim=1, keepdim=True)
        f2a_loss = torch.mean(torch.norm(a_emd_t2-audio_emd/torch.norm(audio_emd, dim=1, keepdim=True), dim=1))
        a2f_loss = torch.mean(torch.norm(f_emd_t2-f_emd/torch.norm(f_emd, dim=1, keepdim=True), dim=1))
        loss = loss + args.alpha_trans*(f2a_loss + a2f_loss)/2

    # Supervised Contrastive Learning
    if args.use_video:
        v_emd_proj = v_proj(v_emd[:, :v_dim])  # v_emd[:, :v_dim]是取该模态前一半作为模态共享特征，然后通过MLP变换，并与真实标签做对比。目标是强化共享特征的学习？
    if args.use_audio:
        a_emd_proj = a_proj(audio_emd[:, :a_dim])
    if args.use_flow:
        f_emd_proj = f_proj(f_emd[:, :f_dim])
    if args.use_video and args.use_flow and args.use_audio:
        emd_proj = torch.stack([v_emd_proj, a_emd_proj, f_emd_proj], dim=1)  # 将三个模态的特征拼接起来，形状为[batch_size, 3, 128]
    elif args.use_video and args.use_flow:
        emd_proj = torch.stack([v_emd_proj, f_emd_proj], dim=1)
    elif args.use_video and args.use_audio:
        emd_proj = torch.stack([v_emd_proj, a_emd_proj], dim=1)
    elif args.use_flow and args.use_audio:
        emd_proj = torch.stack([f_emd_proj, a_emd_proj], dim=1)
    # 当调用 criterion_contrast 的前向函数时：
    # emd_proj 的形状：[batch_size, 3, 128]
    # labels 的形状：[batch_size]
    # 这里的 batch_size 取决于你在训练时设置的批量大小（通常通过 args.bsz 指定）。
    # 需要注意的是，SupConLoss 期望的输入形状是 [bsz, n_views, ...]，其中 n_views 在这里是2，对应于视频、光流、音频3种模态。这种设计允许损失函数在不同模态的特征表示之间进行对比学习
    loss_contrast = criterion_contrast(emd_proj, labels) #
    loss = loss + args.alpha_contrast*loss_contrast
  
    # Feature Splitting with Distance
    loss_e = 0
    num_loss = 0
    if args.use_video:  
        loss_e = loss_e - F.mse_loss(v_emd[:, :v_dim], v_emd[:, v_dim:])   # F.mse_loss(v_emd[:, :v_dim], v_emd[:, v_dim:])计算的是模态专有特征和模态共享特征之间的距离，希望这个距离越大越好
        num_loss = num_loss + 1
    if args.use_audio:
        loss_e = loss_e - F.mse_loss(audio_emd[:, :a_dim], audio_emd[:, a_dim:])
        num_loss = num_loss + 1
    if args.use_flow:
        loss_e = loss_e - F.mse_loss(f_emd[:, :f_dim], f_emd[:, f_dim:])
        num_loss = num_loss + 1
    
    loss = loss + args.explore_loss_coeff * loss_e/num_loss

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predict, loss

def validate_one_step(clip, labels, flow, spectrogram):
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()
    
    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip) 
            v_feat = (x_slow.detach(), x_fast.detach())  

            v_feat = model.module.backbone.get_predict(v_feat)
            predict1, v_emd = model.module.cls_head(v_feat)
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)
            audio_predict, audio_emd = audio_cls_model(audio_feat.detach())
        if args.use_flow:
            f_feat = model_flow.module.backbone.get_feature(flow)  
            f_feat = model_flow.module.backbone.get_predict(f_feat)
            f_predict, f_emd = model_flow.module.cls_head(f_feat)

        if args.use_video and args.use_flow and args.use_audio:
            feat = torch.cat((v_emd, audio_emd, f_emd), dim=1)
        elif args.use_video and args.use_flow:
            feat = torch.cat((v_emd, f_emd), dim=1)
        elif args.use_video and args.use_audio:
            feat = torch.cat((v_emd, audio_emd), dim=1)
        elif args.use_flow and args.use_audio:
            feat = torch.cat((f_emd, audio_emd), dim=1)

        predict = mlp_cls(feat)

    loss = criterion(predict, labels)

    return predict, loss

class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)

# 模态转换的作用，帮助不同模态之间的特征对其和信息交互
class EncoderTrans(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat

# 这个ProjectHead模块的主要作用是将输入特征映射到一个新的特征空间，可能用于特征提取或降维等任务。通过使用多层结构和归一化，它可以帮助学习更有效的特征表示
class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-s','--source_domain', nargs='+', help='<Required> Set source_domain', required=True)
    parser.add_argument('-t','--target_domain', nargs='+', help='<Required> Set target_domain', required=True)
    parser.add_argument('--datapath', type=str, default='/path/to/EPIC-KITCHENS/', help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--bsz', type=int, default=16, help='batch_size')
    parser.add_argument("--nepochs", type=int, default=15)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--alpha_trans', type=float, default=0.1, help='alpha_trans')
    parser.add_argument("--trans_hidden_num", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument('--temp', type=float, default=0.1, help='temp')
    parser.add_argument('--alpha_contrast', type=float, default=3.0, help='alpha_contrast')
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument('--explore_loss_coeff', type=float, default=0.7, help='explore_loss_coeff')#alpha_dis
    parser.add_argument("--BestEpoch", type=int, default=0)
    parser.add_argument('--BestAcc', type=float, default=0, help='BestAcc')
    parser.add_argument('--BestTestAcc', type=float, default=0, help='BestTestAcc')
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_video', action='store_true')
    parser.add_argument('--use_audio', action='store_true')
    parser.add_argument('--use_flow', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    input_dim = 0

    cfg = None
    cfg_flow = None
    
    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)# model本身的模型参数被冻结
        model.cls_head.fc_cls = nn.Linear(2304, 8).cuda()  # 只需要训练线性分类头，最后的预测类别是8个
        cfg = model.cfg
        model = torch.nn.DataParallel(model)# 这行代码的目的是通过利用多 GPU 资源来加速模型的训练过程，本机为单GPU时无效
        # 1152的维度是2304的一半，即这里认为一半是模态专有特性，一半是模态共享特性
        # 这里各设置为一半，是否可以门控网络进行调整？即通过学习进行加权
        # 将不同模态统一到out_dim的维度上
        v_proj = ProjectHead(input_dim=1152, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()  # 进行对比损失时用到，可以看到作者在这里涉及了三种分类头
        input_dim = input_dim + 2304

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, 8).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)

        f_proj = ProjectHead(input_dim=1024, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 2048

    if args.use_audio:
        # audio_model（AVENet）和audio_cls_model（AudioAttGenModule）可能形成了一个两阶段的音频处理流程。AVENet 负责初始的特征提取，而 AudioAttGenModule 负责进一步的特征处理、注意力应用和最终分类。这种配合使用的方式允许模型充分利用预训练的通用音频特征，同时针对特定任务进行优化，提高整体的音频处理和分类性能。

        audio_args = get_arguments()
        # 使用VGGSound的预训练模型
        audio_model = AVENet(audio_args)
        checkpoint = torch.load("pretrained_models/vggsound_avgpool.pth.tar")
        audio_model.load_state_dict(checkpoint['model_state_dict'])  #载入权重
        audio_model = audio_model.cuda()
        audio_model.eval()

        # 训练audio_cls_model和a_proj的参数
        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False) #载入权重，这里选择不进行完全匹配
        audio_cls_model.fc = nn.Linear(512, 8)
        audio_cls_model = audio_cls_model.cuda()

        a_proj = ProjectHead(input_dim=256, hidden_dim=args.hidden_dim, out_dim=args.out_dim).cuda()
        input_dim = input_dim + 512
    # 分类器，输入维度为所有隐藏层（含模态共享、模态专有维度）
    # The EPIC-Kitchens dataset includes eight actions (‘put’,‘take’, ‘open’, ‘close’, ‘wash’, ‘cut’, ‘mix’, and ‘pour’)
    mlp_cls = Encoder(input_dim=input_dim, out_dim=8)  # 单独搭建的预测头，2层MLP
    mlp_cls = mlp_cls.cuda()

    if args.use_video and args.use_flow and args.use_audio:
        mlp_v2f = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=2048).cuda()
        mlp_f2v = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=2304).cuda()
        mlp_v2a = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2v = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2304).cuda()
        mlp_f2a = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2f = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2048).cuda()
    elif args.use_video and args.use_flow:
        mlp_v2f = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=2048).cuda()
        mlp_f2v = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=2304).cuda()
    elif args.use_video and args.use_audio:
        mlp_v2a = EncoderTrans(input_dim=2304, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2v = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2304).cuda()
    elif args.use_flow and args.use_audio:
        mlp_f2a = EncoderTrans(input_dim=2048, hidden=args.trans_hidden_num, out_dim=512).cuda()
        mlp_a2f = EncoderTrans(input_dim=512, hidden=args.trans_hidden_num, out_dim=2048).cuda()


    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path_model = "models/"
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)

    log_name = "log%s2%s"%(args.source_domain, args.target_domain)
    if args.use_video:
        log_name = log_name + '_video'
    if args.use_flow:
        log_name = log_name + '_flow'
    if args.use_audio:
        log_name = log_name + '_audio'
    log_name = log_name + args.appen
    log_path = base_path + log_name + '.csv'
    print(log_path)

    run = wandb.init(
        project="SimMMDG",    # Specify your project
        config={                         # Track hyperparameters and metadata
            "learning_rate": args.lr,
            "epochs": args.nepochs,
        },
    )

    # 交叉熵损失
    criterion = nn.CrossEntropyLoss() 
    criterion = criterion.cuda()
    batch_size = args.bsz
    # 有监督对比损失
    criterion_contrast = SupConLoss(temperature=args.temp)
    criterion_contrast = criterion_contrast.cuda()
    # ------------------------------------------------------------------------------------------------------------
    # 需要训练的参数
    params = list(mlp_cls.parameters())
    if args.use_video:
        # SlowFast网络包含两个路径：
        # Slow路径：捕获空间语义信息
        # Fast路径：捕获运动信息
        params = params + list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) + list(model.module.cls_head.parameters()) + list(v_proj.parameters())
    if args.use_flow:
        # SlowOnly是SlowFast网络的一个变体，只保留了Slow路径。
        # 它专门设计用于处理光流输入，因为光流本身已经包含了运动信息
        params = params + list(model_flow.module.backbone.layer4.parameters()) +list(model_flow.module.cls_head.parameters()) + list(f_proj.parameters())
    if args.use_audio:
        params = params + list(audio_cls_model.parameters()) + list(a_proj.parameters())
    
    if args.use_video and args.use_flow and args.use_audio:
        params = params + list(mlp_v2a.parameters())+list(mlp_a2v.parameters())
        params = params + list(mlp_v2f.parameters())+list(mlp_f2v.parameters())
        params = params + list(mlp_f2a.parameters())+list(mlp_a2f.parameters())
    elif args.use_video and args.use_flow:
        params = params + list(mlp_v2f.parameters())+list(mlp_f2v.parameters())
    elif args.use_video and args.use_audio:
        params = params + list(mlp_v2a.parameters())+list(mlp_a2v.parameters())
    elif args.use_flow and args.use_audio:
        params = params + list(mlp_f2a.parameters())+list(mlp_a2f.parameters())

    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    
    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc

    if args.resumef:
        resume_file = base_path_model + log_name + '.pt'
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch']+1
    
        BestLoss = checkpoint['BestLoss']
        BestEpoch = checkpoint['BestEpoch']
        BestAcc = checkpoint['BestAcc']
        BestTestAcc = checkpoint['BestTestAcc']

        if args.use_video:
            model.load_state_dict(checkpoint['model_state_dict'])
            v_proj.load_state_dict(checkpoint['v_proj_state_dict'])
        if args.use_flow:
            model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
            f_proj.load_state_dict(checkpoint['f_proj_state_dict'])
        if args.use_audio:
            audio_model.load_state_dict(checkpoint['audio_model_state_dict'])
            audio_cls_model.load_state_dict(checkpoint['audio_cls_model_state_dict'])
            a_proj.load_state_dict(checkpoint['a_proj_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        if args.use_video and args.use_flow and args.use_audio:
            mlp_v2a.load_state_dict(checkpoint['mlp_v2a_state_dict'])
            mlp_a2v.load_state_dict(checkpoint['mlp_a2v_state_dict'])
            mlp_v2f.load_state_dict(checkpoint['mlp_v2f_state_dict'])
            mlp_f2v.load_state_dict(checkpoint['mlp_f2v_state_dict'])
            mlp_f2a.load_state_dict(checkpoint['mlp_f2a_state_dict'])
            mlp_a2f.load_state_dict(checkpoint['mlp_a2f_state_dict'])
        elif args.use_video and args.use_flow:
            mlp_v2f.load_state_dict(checkpoint['mlp_v2f_state_dict'])
            mlp_f2v.load_state_dict(checkpoint['mlp_f2v_state_dict'])
        elif args.use_video and args.use_audio:
            mlp_v2a.load_state_dict(checkpoint['mlp_v2a_state_dict'])
            mlp_a2v.load_state_dict(checkpoint['mlp_a2v_state_dict'])
        elif args.use_flow and args.use_audio:
            mlp_f2a.load_state_dict(checkpoint['mlp_f2a_state_dict'])
            mlp_a2f.load_state_dict(checkpoint['mlp_a2f_state_dict'])
        mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
    else:
        print("Training From Scratch ..." )
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)

    train_dataset = EPICDOMAIN(split='train', domain=args.source_domain, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)

    val_dataset = EPICDOMAIN(split='test', domain=args.source_domain, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)

    test_dataset = EPICDOMAIN(split='test', domain=args.target_domain, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath, use_video=args.use_video, use_flow=args.use_flow, use_audio=args.use_audio)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    with open(log_path, "a") as f:
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']:  # 官方进行了标准划分
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                mlp_cls.train(split == 'train')
                if args.use_video:
                    model.train(split == 'train')
                    v_proj.train(split == 'train')
                if args.use_flow:
                    model_flow.train(split == 'train')
                    f_proj.train(split == 'train')
                if args.use_audio:
                    audio_cls_model.train(split == 'train')
                    a_proj.train(split == 'train')
                if args.use_video and args.use_flow and args.use_audio:
                    mlp_v2a.train(split == 'train')
                    mlp_a2v.train(split == 'train')
                    mlp_v2f.train(split == 'train')
                    mlp_f2v.train(split == 'train')
                    mlp_f2a.train(split == 'train')
                    mlp_a2f.train(split == 'train')
                elif args.use_video and args.use_flow:
                    mlp_v2f.train(split == 'train')
                    mlp_f2v.train(split == 'train')
                elif args.use_video and args.use_audio:
                    mlp_v2a.train(split == 'train')
                    mlp_a2v.train(split == 'train')
                elif args.use_flow and args.use_audio:
                    mlp_f2a.train(split == 'train')
                    mlp_a2f.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, flow, spectrogram, labels)) in enumerate(dataloaders[split]):
                        if split=='train':
                            predict1, loss = train_one_step(clip, labels, flow, spectrogram)
                        else:
                            predict1, loss = validate_one_step(clip, labels, flow, spectrogram)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),
                                                                                                  acc / float(count)))
                        if split=='train':
                            wandb.log({"train loss": loss.item(), "train Acc": acc / float(count)})
                        elif split=='val':
                            wandb.log({"val loss": loss.item(), "val Acc": acc / float(count)})
                        elif split=='test':
                            wandb.log({"test loss": loss.item(), "test Acc": acc / float(count)})

                        pbar.update()
                    # 验证时，更新bestacc，bestloss，bestepoch
                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:  # 在验证集的时候进行计算
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)  # 更新验证集上的最佳准确率
                            # 验证时，输出当前epoch，loss，acc，bestepoch，bestloss，bestvalacc
                    # 其实这里应该只有验证集效果更好时，才需要去更新测试集上的最好结果，而不应该在每个epoch都计算并输出test上的结果。
                    if split == 'test':
                        currenttestAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:  # 如果验证集上有改进，同步更新测试集上的最佳准确率
                            BestTestAcc = currenttestAcc  # 这种写法更加科学
                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'optimizer': optim.state_dict(),
                                }
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                                
                                if args.use_video:
                                    save['v_proj_state_dict'] = v_proj.state_dict()
                                    save['model_state_dict'] = model.state_dict()
                                if args.use_flow:
                                    save['f_proj_state_dict'] = f_proj.state_dict()
                                    save['model_flow_state_dict'] = model_flow.state_dict()
                                if args.use_audio:
                                    save['a_proj_state_dict'] = a_proj.state_dict()
                                    save['audio_model_state_dict'] = audio_model.state_dict()
                                    save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                                if args.use_video and args.use_flow and args.use_audio:
                                    save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                    save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                    save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                    save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                    save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                    save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()
                                elif args.use_video and args.use_flow:
                                    save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                    save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                elif args.use_video and args.use_audio:
                                    save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                    save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                elif args.use_flow and args.use_audio:
                                    save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                    save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()

                                torch.save(save, base_path_model + log_name + '_best_%s.pt'%(str(epoch_i)))

                        if args.save_checkpoint:
                            save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'optimizer': optim.state_dict(),
                                }
                            save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                            
                            if args.use_video:
                                save['v_proj_state_dict'] = v_proj.state_dict()
                                save['model_state_dict'] = model.state_dict()
                            if args.use_flow:
                                save['f_proj_state_dict'] = f_proj.state_dict()
                                save['model_flow_state_dict'] = model_flow.state_dict()
                            if args.use_audio:
                                save['a_proj_state_dict'] = a_proj.state_dict()
                                save['audio_model_state_dict'] = audio_model.state_dict()
                                save['audio_cls_model_state_dict'] = audio_cls_model.state_dict()
                            if args.use_video and args.use_flow and args.use_audio:
                                save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                                save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                                save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()
                            elif args.use_video and args.use_flow:
                                save['mlp_v2f_state_dict'] = mlp_v2f.state_dict()
                                save['mlp_f2v_state_dict'] = mlp_f2v.state_dict()
                            elif args.use_video and args.use_audio:
                                save['mlp_v2a_state_dict'] = mlp_v2a.state_dict()
                                save['mlp_a2v_state_dict'] = mlp_a2v.state_dict()
                            elif args.use_flow and args.use_audio:
                                save['mlp_f2a_state_dict'] = mlp_f2a.state_dict()
                                save['mlp_a2f_state_dict'] = mlp_a2f.state_dict()

                            torch.save(save, base_path_model + log_name + '.pt')
                    # train val test都输出loss和acc    
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)
                    #test时输出当前epoch，loss，acc，bestepoch，bestloss，bestvalacc，besttestacc
                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)

    f.close()
