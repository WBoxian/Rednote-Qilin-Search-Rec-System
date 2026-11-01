export interface MetricComparison {
  name: string;
  baseline: number;
  current: number;
  absolute_delta: number;
  absolute_delta_pp: number;
  relative_delta_pct: number | null;
  ci95_delta?: [number, number];
}
export interface ComparisonGroup {
  scope: string;
  baseline_name: string;
  current_name?: string;
  metrics: MetricComparison[];
  diagnostics?: {
    median_candidates?: number;
    requests_at_most_10_fraction?: number;
    original_position_NDCG_at10?: number;
    warning?: string;
  };
}
export interface AblationRow {
  id: string;
  label: string;
  status: 'baseline' | 'accepted' | 'diagnostic_rejected';
  metrics: Record<string, number>;
  changed?: string;
  comparison?: Record<string, { absolute_delta_pp: number; relative_delta_pct: number | null }>;
}
export function metricLabel(name: string): string {
  const labels: Record<string, string> = { HitRate: '命中率', Recall: '召回率', MRR: '平均倒数排名', NDCG: '归一化折损累计增益', AUC: '总体区分能力', GAUC: '用户加权区分能力', MedianFirstHitRank: '首命中中位排名' };
  if (name === 'MedianFirstHitRank') return '首命中中位排名（仅前500条内命中的请求）';
  if (name === 'known_positive_NDCG@10') return '已知点击排序质量（NDCG@10）';
  const [key] = name.split('@');
  return labels[key] ? `${labels[key]}（${name}）` : name;
}
export function signed(value: unknown, digits = 2, suffix = ''): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}${suffix}`;
}
export function deltaClass(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value > 0 ? 'delta-positive' : value < 0 ? 'delta-negative' : ''
    : '';
}
export function statusLabel(status: AblationRow['status']): string {
  return { baseline: '基线', accepted: '采用', diagnostic_rejected: '未采用' }[status] || '诊断';
}

export function evaluationText(value: string): string {
  const labels: Record<string, string> = {
    'Reference before ranking-prior and subsequent recall iterations': '排序先验与后续召回迭代之前的参考版本',
    'Same frozen model and fusion; current-request numerical contract replay': '模型与融合不变，按当前请求推理口径回放',
    'Fixed input pool, ordering=coarse; no model retraining in this contrast': '固定候选池，仅比较粗排顺序，不重训模型',
    'Fixed input pool, ordering=fine; no model retraining in this contrast': '固定候选池，仅比较精排顺序，不重训模型',
    'Fixed input pool, ordering=final; no model retraining in this contrast': '固定候选池，比较最终融合顺序，不重训模型',
    'Frozen validation-selected reranker; rejected by full-test release guard': '验证集选出的排序模型未通过全量测试发布门槛',
    'Three extra train-only epochs; A/B/C duration-matched controls; none passed validation': '仅用训练集追加三轮，统一训练时长对照，均未通过验证集验收',
    'Same v3 weights and all-past graph; history filtering and fixed context fusion controls': '固定第三版模型与历史协同图，对照历史过滤和上下文融合',
    'Same learned weights and CF graph; train-only PCA residual representation; fixed 256-dimensional ANN': '固定模型与协同图，用训练集拟合内容残差，近邻检索维度固定为256',
    'Same residual model and CF graph; validation selected ANN2/Swing1/UserCF1 from eight fixed trials': '固定残差模型与协同图，验证集从八组权重中选出2比1比1融合',
    'Uniform random ordering expectation': '均匀随机排序期望',
    'Random expected ordering': '随机排序期望',
    'All rows retain the complete official test cohort. Rejected diagnostics are not served. A/B/C share extra training duration; graph refit data-volume effects are reported separately. Historical test was previously observed. Canonical Search stages share one regenerated input pool; the frozen baseline retains its recorded numerical contract.': '所有实验保留完整官方测试请求；未通过验收的方案不启用。追加训练采用相同时长对照，协同图数据量变化单独报告。测试集已被多轮观察。搜索阶段对比使用同一候选池，冻结基线保留原推理口径。',
    'Frozen DSSM v3 + original fusion': '冻结基线：DSSM第三版与原融合',
    'Current released ordering': '当前服务顺序',
    'Frozen v3 recall order': '冻结基线召回顺序',
    'Canonical request recall order': '统一请求口径的召回顺序',
    'Coarse only': '仅粗排', 'Coarse + fine': '粗排与精排', 'Fine with recall prior': '精排融合召回先验',
    'Ranking-prior experiment': '排序先验实验', 'continuation_control_epoch_6': '追加训练对照（第六轮）',
    'masked_inbatch_epoch_6': '屏蔽潜在误负样本（第六轮）', 'masked_logged_hard_epoch_6': '加入曝光难负样本（第六轮）',
    'filter_history': '过滤历史已消费内容', 'filter_history_context': '过滤历史并调整上下文',
    'DSSM residual + history filter': 'DSSM内容残差与历史过滤',
    'Residual DSSM + validation-selected fusion': '内容残差与验证集选择的融合',
    'Uniform random expected order': '均匀随机排序期望',
  };
  return labels[value] || value;
}
