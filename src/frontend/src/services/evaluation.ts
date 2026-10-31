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
  return name === 'known_positive_NDCG@10' ? '已知点击 NDCG@10' : name;
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
