import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------
# 测试其他的距离度量损失
# 使用KL散度计算视频模态的模态专有特征和模态共享特征之间损失
v_emd=torch.randn(16,2304)
v_dim = 1152
v_shared = v_emd[:, :v_dim]
v_specific = v_emd[:, v_dim:]
v_specific_log=torch.nn.functional.log_softmax(v_specific,dim=1)#先转化为概率，再取对数
v_shared = torch.nn.functional.softmax(v_shared,dim=1)
KL_loss= torch.nn.KLDivLoss(reduce="batchmean")
loss_dis_v=KL_loss(v_specific_log, v_shared)
print(loss_dis_v)



# --------------------------------------------------------------------------------------------
# 使用JS散度(jensen-shannon divergence)计算视频模态的模态专有特征和模态共享特征之间损失
# v_emd=torch.randn(16,2304)
# v_dim = 1152
# v_shared = v_emd[:, :v_dim]
# v_specific = v_emd[:, v_dim:]
# v_specific_log=torch.nn.functional.log_softmax(v_specific,dim=1)#先转化为概率，再取对数
# v_shared = torch.nn.functional.softmax(v_shared,dim=1)
# # 使用JS散度(jensen-shannon divergence)计算损失
# js_loss = torch.nn.functional.js_div(v_specific_log, v_shared)
# print(js_loss)


# --------------------------------------------------------------------------------------------
# 使用对比损失计算模态内部专有特征的损失
# class ProjectHead(nn.Module):
#     def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
#         super(ProjectHead, self).__init__()
#         self.head = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(hidden_dim, out_dim)
#             )
        
#     def forward(self, feat):
#         feat = F.normalize(self.head(feat), dim=1)
#         return feat

# class ModalitySpecificContrastive(nn.Module):
#     def __init__(self, temp=0.07):
#         super().__init__()
#         self.temp = temp
        
#     def forward(self, specific_feat, labels):
#         # 对每个模态的specific特征进行对比学习
#         # 一个模态内同类样本的specific特征应该相似
#         specific_feat = F.normalize(specific_feat, dim=1)#对输入的特征向量进行L2归一化，确保所有特征向量的长度为1
#         sim_matrix = torch.matmul(specific_feat, specific_feat.T) / self.temp#通过矩阵乘法计算所有样本对之间的余弦相似度
#         labels = labels.view(-1, 1)
#         mask = torch.eq(labels, labels.T).float()#创建掩码矩阵，相同类别的样本对应位置为1，不同类别为0
        
#         # 计算对比损失
#         logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)#进行数值稳定性处理，减去每行的最大值
#         sim_matrix = sim_matrix - logits_max.detach()#进行数值稳定性处理，减去每行的最大值
#         exp_sim = torch.exp(sim_matrix)
#         log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))#实际上是在计算 softmax 的对数形式，使用对数形式可以避免指数计算时的数值问题
#         mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)#只考虑相同类别的样本对的概率
#         loss = -mean_log_prob.mean()#取负值并求平均得到最终损失
#         return loss

# v_emd=torch.randn(16,2304)
# v_dim = 1152
# labels=torch.randint(0, 8, (16,))
# v_specific = v_emd[:, v_dim:]
# a_emd=torch.randn(16,2048)
# a_dim = 1024
# a_specific = a_emd[:, a_dim:]
# f_emd=torch.randn(16,512)
# f_dim = 256
# f_specific = f_emd[:, f_dim:]

# v_proj = ProjectHead(input_dim=v_dim, hidden_dim=2408, out_dim=128)  # 进行对比损失时用到
# f_proj = ProjectHead(input_dim=f_dim, hidden_dim=2408, out_dim=128)
# a_proj = ProjectHead(input_dim=a_dim, hidden_dim=2408, out_dim=128)

# v_specific_proj = v_proj(v_specific)
# a_specific_proj = a_proj(a_specific)
# f_specific_proj = f_proj(f_specific)

# specific_contrast = ModalitySpecificContrastive()
# specific_contrast_loss = (specific_contrast(v_specific_proj, labels) + 
#                          specific_contrast(a_specific_proj, labels) + 
#                          specific_contrast(f_specific_proj, labels)) / 3
# print(specific_contrast_loss)

# --------------------------------------------------------------------------------------------
# **模态特定的预测任务**：
# 辅助分类头，用于预测模态的类别
# 实验结果分析：不到十分之一个epoch,specific_pred_loss就降到了0.01以下，说明模态特定的预测任务效果很好，但是对整体分类任务的监督估计不会起作用
# class ModalityPredictor(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.predictor = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 3)  # 预测类别或其他任务
#         )
    
#     def forward(self, specific_feat):
#         return self.predictor(specific_feat)

# v_emd=torch.randn(16,2304)
# v_dim = 1152
# # v_label的类别是1
# v_label = torch.ones(16).long()  # 添加一个维度
# v_specific = v_emd[:, v_dim:]

# a_emd=torch.randn(16,2048)
# a_dim = 1024
# # a_label的类别是2
# a_label = torch.ones(16).long()
# a_specific = a_emd[:, a_dim:]

# f_emd=torch.randn(16,512)
# f_dim = 256
# # f_label的类别是3
# f_label = torch.ones(16).long()
# f_specific = f_emd[:, f_dim:]

# v_predictor = ModalityPredictor(v_dim)
# a_predictor = ModalityPredictor(a_dim)
# f_predictor = ModalityPredictor(f_dim)
# criterion=nn.CrossEntropyLoss()
# specific_pred_loss = (criterion(v_predictor(v_specific), v_label) +
#                      criterion(a_predictor(a_specific), a_label) +
#                      criterion(f_predictor(f_specific), f_label)) / 3
# print(specific_pred_loss)

# --------------------------------------------------------------------------------------------
# 使用互信息计算模态内部专有特征的损失
# 代码的目的
# 特征表示学习：通过最大化同一模态内的互信息，学习更好的特征表示
# 自监督学习：不需要标签，通过数据本身构建学习信号
# 模态内一致性：确保同一模态的特征具有内部一致性
# def compute_mutual_info(x1, x2):
#     # 计算两个特征之间的互信息
#     joint = torch.cat([x1, x2], dim=1)#将两个特征直接连接，代表正样本对
#     joint = F.normalize(joint, dim=1)
#     marginal = torch.cat([x1, x2[torch.randperm(x2.size(0))]], dim=1)#通过随机打乱第二个特征的顺序创建负样本对
#     marginal = F.normalize(marginal, dim=1)
    
#     # 最大化同模态内的互信息，最小化不同模态间的互信息
#     # 具体任务可能需要微调：如果发现学习不稳定，可以适当增大；如果需要更强的区分性，可以适当减小
#     pos_sim = torch.exp(torch.sum(joint * joint, dim=1) / 0.07)#0.07是温度参数，用于调节分布的平滑程度
#     neg_sim = torch.exp(torch.sum(marginal * marginal, dim=1) / 0.07)
    
#     return -torch.log(pos_sim / (pos_sim + neg_sim)).mean()#使用InfoNCE损失的形式
# # 目标是最大化正样本对的相似度，最小化负样本对的相似度

# v_emd=torch.randn(16,2304)
# v_dim = 1152
# labels=torch.randint(0, 8, (16,))
# v_specific = v_emd[:, v_dim:]
# a_emd=torch.randn(16,2048)
# a_dim = 1024
# a_specific = a_emd[:, a_dim:]
# f_emd=torch.randn(16,512)
# f_dim = 256
# f_specific = f_emd[:, f_dim:]

# # 
# intra_modal_mi_loss = (compute_mutual_info(v_emd[:, :v_dim], v_emd[:, v_dim:]) +
#                       compute_mutual_info(a_emd[:, :a_dim], a_emd[:, a_dim:]) +
#                       compute_mutual_info(f_emd[:, :f_dim], f_emd[:, f_dim:])) / 3
# print(intra_modal_mi_loss)

#   ----------------------------------------------------------------------------------
# 测试其他的距离度量损失
# 使用KL散度计算视频模态的模态专有特征和模态共享特征之间损失
# KL散度大，说明模态专有特征和模态共享特征的分布差异大，即两者“越不像”，因此这里的计算结果取负号
#     KL_loss= torch.nn.KLDivLoss(reduce="batchmean")
#     v_specific = v_emd[:, v_dim:]
#     v_specific_log=torch.nn.functional.log_softmax(v_specific,dim=1)#先转化为概率，再取对数
#     v_shared = v_emd[:, :v_dim]
#     v_shared = torch.nn.functional.softmax(v_shared,dim=1)
#     loss_dis_v=-KL_loss(v_specific_log, v_shared)

#     f_specific = f_emd[:, f_dim:]
#     f_specific_log=torch.nn.functional.log_softmax(f_specific,dim=1)#先转化为概率，再取对数
#     f_shared = f_emd[:, :f_dim]
#     f_shared = torch.nn.functional.softmax(f_shared,dim=1)
#     loss_dis_f=-KL_loss(f_specific_log, f_shared)

#     a_specific = audio_emd[:, a_dim:]
#     a_specific_log=torch.nn.functional.log_softmax(a_specific,dim=1)#先转化为概率，再取对数
#     a_shared = audio_emd[:, :a_dim]
#     a_shared = torch.nn.functional.softmax(a_shared,dim=1)
#     loss_dis_a=-KL_loss(a_specific_log, a_shared)

#     loss_e=0
#     loss_e=loss_dis_v+loss_dis_f+loss_dis_a
#     num_loss=3
#     wandb.log({"distance loss": (loss_e/num_loss).item()})
#     loss = loss + 100 * loss_e/num_loss
# --------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
    # 使用互信息计算模态内部专有特征的损失
    # v_specific = v_emd[:, v_dim:]
    # a_specific = audio_emd[:, a_dim:]
    # f_specific = f_emd[:, f_dim:]
    # v_specific_proj = v_specific_proj_head(v_specific)
    # a_specific_proj = a_specific_proj_head(a_specific)
    # f_specific_proj = f_specific_proj_head(f_specific)
    # intra_modal_mi_loss = (compute_mutual_info(v_specific_proj, v_shared_emd_proj) +\
    #                     compute_mutual_info(a_specific_proj, a_shared_emd_proj) +\
    #                     compute_mutual_info(f_specific_proj, f_shared_emd_proj)) / 3
    # loss = loss + 0.5 * intra_modal_mi_loss
    # wandb.log({"intra_modal_mi_loss": intra_modal_mi_loss.item()})
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
    # 使用模态特定的预测任务
    # v_label = torch.zeros(16).long().cuda()  # 添加一个维度
    # a_label = (torch.ones(16)).long().cuda()
    # f_label = (torch.ones(16)*2).long().cuda()
    # modal_pred_criterion=nn.CrossEntropyLoss()
    # specific_modal_pred_loss = (modal_pred_criterion(v_modal_predictor(v_emd[:, v_dim:]), v_label) +
    #                     modal_pred_criterion(a_modal_predictor(audio_emd[:, a_dim:]), a_label) +
    #                     modal_pred_criterion(f_modal_predictor(f_emd[:, f_dim:]), f_label)) / 3
    # loss = loss + 0.5*specific_modal_pred_loss
    # wandb.log({"specific_modal_pred_loss": specific_modal_pred_loss.item()})
# ---------------------------------------------------------------------------