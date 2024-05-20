'''
This implementation is from https://github.com/HobbitLong/SupContrast with small modifications.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class supervised_contrastive_loss(nn.Module):
    """
    Supervised Contrastive Loss implementation.
    
    :param z: Embeddings tensor with shape (2M, embedding_dim), where 2M is the total number of positive and negative samples.
    :param labels: Labels tensor with shape (2M,), corresponding to the embeddings.
    :param temperature: Temperature parameter.
    :param delta: Hyperparameter controlling the labeling effect.
    :return: Loss value.
    """
    def __init__(self, temperature=0.2, batch_size=128, label_ratio=0.1):
        super(supervised_contrastive_loss, self).__init__()
        self.temperature = temperature
        self.batch_size = int(batch_size * label_ratio)


    def forward(self, features, labels, delta=1.0):
        # Normalize features
        features = F.normalize(features, p=2, dim=1)# 使用L2范数对特征进行归一化，这有助于稳定训练过程并提高性能
        
        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.T) / self.temperature #计算特征之间的点积并除以温度参数
    
        label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0)).mean(dim=(-2, -1))#y1-y2的绝对值，为了调整由于标签引起的相似度
        # print(label_diff)
        sim_matrix -= delta * label_diff
    
        # Mask-out self-similarities
        batch_size = features.size(0)
        mask_self_similarity = torch.eye(batch_size, dtype=torch.bool).cuda()
        sim_matrix.masked_fill_(mask_self_similarity, 0)#为避免特征与自身的比较，通过一个掩码矩阵将sim_matrix对角线上的元素置0。
    
        # Compute loss
        sim_pos = torch.sum(features[:self.batch_size] * features[self.batch_size:], dim=-1) / self.temperature
        sim_pos = torch.cat([sim_pos, sim_pos], dim=0)
        loss = -torch.log(torch.exp(sim_pos) / torch.exp(sim_matrix).sum(dim=1))
    
        return loss.mean()



class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""
	def __init__(self, temperature=0.07):
		super(SupConLoss, self).__init__()
		self.temperature = temperature


	def forward(self, features, labels=None, mask=None):
		"""Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf
		Args:
		features: hidden vector of shape [bsz, n_views, ...].形状的隐藏向量[bsz，n_views，…]。
		labels: ground truth of shape [bsz]. 形状的基本真值
		mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
		has the same class as sample i. Can be asymmetric.
		mask:形状对比掩模[bsz，bsz]，掩模{i，j}=1，如果样本j与样本i具有相同的类。i,j可以是不对称的。
		Returns:
		A loss scalar.损失标量
		"""
		device = (torch.device('cuda')
			if features.is_cuda
			else torch.device('cpu'))

		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		batch_size = features.shape[0]

		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(device)

		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)

			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')

			mask = torch.eq(labels, labels.T).float().to(device)
		else:
			mask = mask.float().to(device)

		contrast_count = features.shape[1]
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
		anchor_feature = contrast_feature
		anchor_count = contrast_count
		anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()
		mask = mask.repeat(anchor_count, contrast_count)

		# mask-out self-contrast cases
		logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)

		mask = mask * logits_mask                                                    #

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask
		eps = 1e-30
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+eps)

		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+eps)

		# loss
		loss = -  mean_log_prob_pos


		loss = loss.view(anchor_count, batch_size).mean()
		return loss 
