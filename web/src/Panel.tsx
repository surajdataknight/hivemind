// the scenario panel: what hivemind is doing on each case, as it happens (plan, checks, findings, fact sheet).
// read only; it asks the agents service every 2 s. it needs the panel key, kept in this tab only.

import { useEffect, useState, type FormEvent } from "react";
import { ApiError } from "./api";
import {
  byRevision,
  listCases,
  loadCase,
  statusOf,
  waves,
  type CaseDetail,
  type CaseRow,
  type Finding,
  type Task,
} from "./panelData";

const KEY = "hivemind.panelKey";
const EVERY_MS = 2000;

const words = (value: string) => value.replace(/_/g, " ");
const time = (iso: string) => new Date(iso).toLocaleTimeString();

function savedKey(): string {
  try {
    return sessionStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
}

export function Panel() {
  const [key, setKey] = useState(savedKey);
  const [typed, setTyped] = useState("");
  const [cases, setCases] = useState<CaseRow[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!key) return;
    let stopped = false;
    const tick = async () => {
      try {
        const rows = await listCases(key);
        if (stopped) return;
        setCases(rows);
        const pick = chosen ?? rows[0]?.case_id ?? null;
        if (pick !== chosen) setChosen(pick);
        setDetail(pick ? await loadCase(key, pick) : null);
        setError(null);
      } catch (exc) {
        if (stopped) return;
        if (exc instanceof ApiError && exc.status === 401) {
          sessionStorage.removeItem(KEY);
          setKey("");
        }
        setError(exc instanceof Error ? exc.message : "the panel could not reach the agents service");
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), EVERY_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [key, chosen]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const value = typed.trim();
    if (!value) return;
    sessionStorage.setItem(KEY, value);
    setKey(value);
    setTyped("");
  };

  if (!key) {
    return (
      <main className="panel-key">
        <h1>Scenario panel</h1>
        <p className="muted">What HiveMind does on each case. Enter the panel key (HM_PANEL_KEY in .env).</p>
        <form onSubmit={submit}>
          <input
            type="password"
            value={typed}
            onChange={(event) => setTyped(event.target.value)}
            aria-label="Panel key"
            autoComplete="off"
          />
          <button type="submit">Open</button>
        </form>
        {error && <p className="error">{error}</p>}
      </main>
    );
  }

  return (
    <main className="panel">
      <aside className="case-list">
        <header>
          <h1>Scenario panel</h1>
          <button
            type="button"
            onClick={() => {
              sessionStorage.removeItem(KEY);
              setKey("");
            }}
          >
            Lock
          </button>
        </header>
        {error && <p className="error">{error}</p>}
        {cases.length === 0 && <p className="muted">No case yet. Report a problem in the chat to start one.</p>}
        {cases.map((row) => (
          <button
            key={row.case_id}
            type="button"
            className={`case-row${row.case_id === chosen ? " chosen" : ""}`}
            onClick={() => setChosen(row.case_id)}
          >
            <strong>{words(row.case_type)}</strong>
            <span className="muted">
              {row.case_id} · {row.order_id ?? "no order"} · {time(row.updated_at)}
            </span>
            <span>
              {row.running ? <span className="tag running">working</span> : null}
              {row.verdict ? <span className={`tag ${row.verdict}`}>{words(row.verdict)}</span> : null}
              <span className="muted"> {row.findings} findings</span>
            </span>
          </button>
        ))}
      </aside>
      <section className="case-detail">{detail ? <CaseView detail={detail} /> : null}</section>
    </main>
  );
}

function CaseView({ detail }: { detail: CaseDetail }) {
  const run = detail.runs[detail.runs.length - 1];
  const byTask = new Map<string, Finding[]>();
  for (const finding of detail.findings.filter((f) => f.fence === run.fence)) {
    byTask.set(finding.task_id, [...(byTask.get(finding.task_id) ?? []), finding]);
  }
  const labels = new Map(detail.findings.map((f, i) => [f.finding_id, `F${i + 1}`]));
  const sheets = detail.fact_sheets.filter((s) => s.fence === run.fence);
  const { first, revised } = byRevision(run.plan.tasks);
  return (
    <>
      <header>
        <h2>
          {words(detail.case_type)} <span className="muted">{detail.case_id}</span>
        </h2>
        <p className="muted">
          order {detail.order_id ?? "none"} · customer {detail.customer_id} · run {run.fence} · plan from{" "}
          {run.plan.source}
          {run.plan.replans ? `, revised ${run.plan.replans}×` : ""}
          {detail.running ? " · working now" : ""}
        </p>
        {run.plan.note && <p className="muted">{run.plan.note}</p>}
      </header>

      <h3>Plan</h3>
      <Waves tasks={first} run={run} running={detail.running} byTask={byTask} labels={labels} />
      {revised.length > 0 && (
        <>
          <h3>
            Added when the plan was revised{" "}
            <span className="muted">(the reconciler asked for more; these ran after the first fact sheet)</span>
          </h3>
          <Waves tasks={revised} run={run} running={detail.running} byTask={byTask} labels={labels} />
        </>
      )}
      {run.plan.dropped.length > 0 && (
        <p className="muted">Refused by the validator: {run.plan.dropped.map(([what, why]) => `${what} (${why})`).join("; ")}</p>
      )}

      {sheets.map(({ round, sheet }) => (
        <div className="sheet" key={round}>
          <h3>
            Fact sheet, round {round} <span className={`tag ${sheet.verdict}`}>{words(sheet.verdict)}</span>{" "}
            <span className="muted">{sheet.source === "code" ? "made by code (no model answer)" : "reconciler"}</span>
          </h3>
          {sheet.note && <p className="muted">{sheet.note}</p>}
          {sheet.claims.length > 0 && (
            <ul>
              {sheet.claims.map((c, i) => (
                <li key={i}>
                  {c.about}: <span className={`tag ${c.status}`}>{c.status}</span>{" "}
                  <span className="muted">{c.finding_ids.map((id) => labels.get(id) ?? id).join(", ")}</span>
                </li>
              ))}
            </ul>
          )}
          <ul>
            {sheet.facts.map((fact, i) => (
              <li key={i}>
                {fact.statement}{" "}
                <span className="muted">{fact.finding_ids.map((id) => labels.get(id) ?? id).join(", ")}</span>
              </li>
            ))}
          </ul>
          {sheet.cross_checks.length > 0 && (
            <>
              <p className="muted">Checked by code:</p>
              <ul>
                {sheet.cross_checks.map((row, i) => (
                  <li key={i}>
                    {row.about}: {words(row.signal)}{" "}
                    <span className={`tag ${row.direction}`}>{row.direction}</span>
                    <span className="muted">{row.finding_ids.map((id) => labels.get(id) ?? id).join(", ")}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
          {sheet.contradictions.map((c, i) => (
            <p className="warn" key={i}>
              Contradiction ({c.finding_ids.map((id) => labels.get(id) ?? id).join(" vs ")}): {c.why}
            </p>
          ))}
          {sheet.gaps.length > 0 && <p className="muted">Gaps: {sheet.gaps.join("; ")}</p>}
          {sheet.follow_ups.length > 0 && <p className="muted">Asked the case lead: {sheet.follow_ups.join("; ")}</p>}
          {sheet.dropped.length > 0 && <p className="muted">Dropped from the model's answer: {sheet.dropped.join("; ")}</p>}
        </div>
      ))}

      <h3>Reads ({detail.reads.length})</h3>
      <p className="muted reads">
        {detail.reads.map((r) => `${r.tool} ${time(r.created_at)}`).join(" · ") || "none"}
      </p>
    </>
  );
}

function Waves({ tasks, run, running, byTask, labels }: {
  tasks: Task[];
  run: CaseDetail["runs"][number];
  running: boolean;
  byTask: Map<string, Finding[]>;
  labels: Map<string, string>;
}) {
  return (
    <div className="waves">
      {waves(tasks).map((wave, i) => (
        <div className="wave" key={i}>
          <span className="muted">wave {i + 1}</span>
          {wave.map((task) => {
            const status = statusOf(task, run.outcomes, running);
            const outcome = run.outcomes[task.task_id];
            return (
              <div className={`task ${status}`} key={task.task_id}>
                <strong>{words(task.capability)}</strong>
                <span className="muted">
                  {task.task_id}
                  {task.mandatory ? " · always runs" : ` · asked by ${task.requested_by}`}
                </span>
                <span className={`tag ${status}`}>{words(status)}</span>
                {outcome?.error && <span className="muted">{outcome.error}</span>}
                {(byTask.get(task.task_id) ?? []).map((f) => (
                  <span className="finding" key={f.finding_id} title={f.evidence_refs.join(", ")}>
                    <b>{labels.get(f.finding_id)}</b> {f.claim}{" "}
                    <span className="muted">
                      ({f.confidence}, {f.evidence_refs.length} read{f.evidence_refs.length === 1 ? "" : "s"})
                    </span>
                  </span>
                ))}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
