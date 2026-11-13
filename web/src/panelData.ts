// the scenario panel's data: what the agents service's read-only panel api returns, and how a plan is laid out

import { ApiError } from "./api";

export type CaseRow = {
  case_id: string;
  case_type: string;
  order_id: string | null;
  customer_id: string;
  updated_at: string;
  running: boolean;
  findings: number;
  verdict: string | null;
};

export type Task = {
  task_id: string;
  capability: string;
  depends_on: string[];
  mandatory: boolean;
  reason: string;
  requested_by: string;
  depth: number;
};

export type Outcome = { task_id: string; status: string; error: string | null; attempts: number };

export type Run = {
  fence: number;
  plan: { source: string; tasks: Task[]; dropped: [string, string][]; missing: string[]; note: string; replans: number };
  outcomes: Record<string, Outcome>;
  updated_at: string;
};

export type Finding = {
  finding_id: string;
  task_id: string;
  capability: string;
  type: string;
  claim: string;
  evidence_refs: string[];
  confidence: string;
  fence: number;
};

export type Sheet = {
  round: number;
  verdict: string;
  source: string;
  facts: { statement: string; finding_ids: string[]; evidence_refs: string[] }[];
  claims: { about: string; sku: string | null; status: string; finding_ids: string[] }[];
  contradictions: { finding_ids: string[]; why: string }[];
  gaps: string[];
  follow_ups: string[];
  cross_checks: { about: string; signal: string; direction: string; finding_ids: string[] }[];
  note: string;
  dropped: string[];
};

export type CaseDetail = {
  case_id: string;
  case_type: string;
  order_id: string | null;
  customer_id: string;
  running: boolean;
  runs: Run[];
  findings: Finding[];
  fact_sheets: { fence: number; round: number; verdict: string; sheet: Sheet }[];
  reads: { evidence_id: string; tool: string; created_at: string }[];
};

async function get<T>(path: string, key: string): Promise<T> {
  const response = await fetch(`/api${path}`, { headers: { Authorization: `Bearer ${key}` } });
  if (!response.ok) {
    let detail = `request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // not json; the status line is all there is
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const listCases = async (key: string) => (await get<{ cases: CaseRow[] }>("/v1/panel/cases", key)).cases;
export const loadCase = (key: string, caseId: string) =>
  get<CaseDetail>(`/v1/panel/cases/${encodeURIComponent(caseId)}`, key);

// the plan in waves: a check sits one wave after the latest check it needs (the order the scheduler can run them in)
export function waves(tasks: Task[]): Task[][] {
  const byId = new Map(tasks.map((t) => [t.task_id, t]));
  const level = new Map<string, number>();
  const levelOf = (task: Task, seen: Set<string>): number => {
    const known = level.get(task.task_id);
    if (known !== undefined) return known;
    if (seen.has(task.task_id)) return 0; // a loop cannot pass the validator; drawn flat if it ever did
    seen.add(task.task_id);
    const needs = task.depends_on.map((id) => byId.get(id)).filter((t): t is Task => t !== undefined);
    const value = needs.length ? Math.max(...needs.map((t) => levelOf(t, seen))) + 1 : 0;
    level.set(task.task_id, value);
    return value;
  };
  const out: Task[][] = [];
  for (const task of tasks) {
    const at = levelOf(task, new Set());
    (out[at] ??= []).push(task);
  }
  return out.filter(Boolean);
}

// the plan split by when its checks came in: the first plan (with what its checks asked for), then the checks the
// case lead added when the reconciler asked for more (with what those asked for). a revision's checks run only after
// the first fact sheet, so they are drawn after the first plan, not in its waves.
export function byRevision(tasks: Task[]): { first: Task[]; revised: Task[] } {
  const revisedIds = new Set<string>();
  for (const task of tasks) {
    if (task.requested_by === "replan" || revisedIds.has(task.requested_by)) revisedIds.add(task.task_id);
  }
  return {
    first: tasks.filter((t) => !revisedIds.has(t.task_id)),
    revised: tasks.filter((t) => revisedIds.has(t.task_id)),
  };
}

// a check with no outcome yet is waiting, or running when the case is being worked and all it needs is done
export function statusOf(task: Task, outcomes: Record<string, Outcome>, running: boolean): string {
  const outcome = outcomes[task.task_id];
  if (outcome) return outcome.status;
  if (!running) return "waiting";
  return task.depends_on.every((id) => outcomes[id]?.status === "done") ? "running" : "waiting";
}
