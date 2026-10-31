"""Request-balanced classification and top-k ranking losses."""
import torch
import torch.nn.functional as F

def balanced_losses(scores,labels,groups,topk=10):
    """Macro request BCE and top-k LambdaRank; delta-NDCG weights are detached."""
    scores=scores.float();labels=labels.float()
    if scores.ndim!=1 or labels.shape!=scores.shape or groups.shape!=scores.shape:
        raise ValueError('One score, binary label and request ID required per exposure')
    if not torch.isfinite(scores).all() or not ((labels==0)|(labels==1)).all():
        raise ValueError('Finite scores and binary labels required')
    _,inverse,counts=torch.unique(groups,return_inverse=True,return_counts=True)
    n_groups=len(counts)
    per_row=F.binary_cross_entropy_with_logits(scores,labels,reduction='none')
    sums=torch.zeros(n_groups,device=scores.device).index_add_(0,inverse,per_row)
    bce=(sums/counts).mean()
    same=groups[:,None]==groups[None,:]
    with torch.no_grad():
        detached=scores.detach();index=torch.arange(len(scores),device=scores.device)
        higher=(detached[None,:]>detached[:,None])|((detached[None,:]==detached[:,None])&(index[None,:]<index[:,None]))
        ranks=1+(same&higher).sum(dim=1)
        discount=torch.where(ranks<=topk,1./torch.log2(ranks.float()+1.),0.)
        positives=torch.zeros(n_groups,device=scores.device).index_add_(0,inverse,labels)
        ideal_table=torch.cat((torch.zeros(1,device=scores.device),
            torch.cumsum(1./torch.log2(torch.arange(2,topk+2,device=scores.device,dtype=torch.float32)),0)))
        ideal=ideal_table[positives.long().clamp(max=topk)]
        valid=same&(labels[:,None]>labels[None,:])
        weights=valid.float()*torch.abs(discount[:,None]-discount[None,:])/ideal[inverse,None].clamp_min(1e-8)
    pair=F.softplus(-(scores[:,None]-scores[None,:]))
    row_loss=(pair*weights).sum(dim=1);row_weight=weights.sum(dim=1)
    group_loss=torch.zeros(n_groups,device=scores.device).index_add_(0,inverse,row_loss)
    group_weight=torch.zeros(n_groups,device=scores.device).index_add_(0,inverse,row_weight)
    eligible=group_weight>0
    rank_loss=(group_loss[eligible]/group_weight[eligible]).mean() if eligible.any() else scores.sum()*0.
    return bce,rank_loss


