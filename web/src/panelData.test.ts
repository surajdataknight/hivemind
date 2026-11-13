import { describe, expect, it } from "vitest";
import { byRevision, statusOf, waves, type Task } from "./panelData";

const task = (id: string, needs: string[] = []): Task => ({
  task_id: id,
  capability: id,
  depends_on: needs,
  mandatory: true,
  reason: "",
  requested_by: "minimum",
  depth: 1,
});

describe("the plan in waves", () => {
  it("puts a check one wave after the latest check it needs", () => {
    const plan = [task("m1"), task("m2"), task("m4"), task("m5", ["m4", "m1"]), task("t1", ["m5"])];
    expect(waves(plan).map((w) => w.map((t) => t.task_id))).toEqual([["m1", "m2", "m4"], ["m5"], ["t1"]]);
  });

  it("draws a loop flat instead of hanging", () => {
    expect(waves([task("a", ["b"]), task("b", ["a"])]).flat()).toHaveLength(2);
  });
});

describe("the plan by revision", () => {
  it("draws the checks a revision added, and what they asked for, after the first plan", () => {
    const plan = [
      task("m1"),
      { ...task("t1"), mandatory: false, requested_by: "case_lead" },
      { ...task("r1", ["m1"]), mandatory: false, requested_by: "m1" },
      { ...task("t2"), mandatory: false, requested_by: "replan" },
      { ...task("r2"), mandatory: false, requested_by: "t2" },
    ];
    const { first, revised } = byRevision(plan);
    expect(first.map((t) => t.task_id)).toEqual(["m1", "t1", "r1"]);
    expect(revised.map((t) => t.task_id)).toEqual(["t2", "r2"]);
  });

  it("leaves a plan never revised whole", () => {
    expect(byRevision([task("m1"), task("m2")]).revised).toEqual([]);
  });
});

describe("a check's status", () => {
  const outcomes = { m1: { task_id: "m1", status: "done", error: null, attempts: 1 } };
  it("is its outcome once it has one", () => {
    expect(statusOf(task("m1"), outcomes, true)).toBe("done");
  });
  it("is running while the case is worked and all it needs is done; else waiting", () => {
    expect(statusOf(task("m5", ["m1"]), outcomes, true)).toBe("running");
    expect(statusOf(task("m6", ["m2"]), outcomes, true)).toBe("waiting");
    expect(statusOf(task("m5", ["m1"]), outcomes, false)).toBe("waiting");
  });
});
