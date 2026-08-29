# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""P70: the results overview's task names open a task page across all experiments.

The reviewer: "you can see all of the tasks being listed here, but you can't click on the
tasks … you see all of the episodes and, obviously, which policy they were on."

`joinTaskEpisodes` and `summarisePolicies` are pure, so they are extracted from
app.js and exercised under node with synthetic fixtures. Verified once against the
live dashboard on the whole corpus: 138 tasks, 1056 episodes, zero count/success
mismatches against /api/overview and zero rows with an unresolved policy.
"""
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "dashboard" / "static" / "app.js"
node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _pure_functions() -> str:
    src = APP_JS.read_text()
    start = src.index("function joinTaskEpisodes")
    end = src.index("async function ensureOverview")
    return src[start:end]


def _run(script: str):
    out = subprocess.run(["node", "--input-type=module", "-e", _pure_functions() + script],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@node
def test_join_pairs_every_episode_with_its_run_policy():
    got = _run(textwrap.dedent("""
        const taskRow = { task: 'T', runs: ['a', 'b'] };
        const runs = [{ run_id: 'a', policy: 'pi05' }, { run_id: 'b', policy: 'cosmos3' }];
        const eps = {
          a: [{ env_id: 0, run_index: 0, success: true, score: 1.0, duration: 3 }],
          b: [{ env_id: 0, run_index: 0, success: false, score: 0.5, duration: 4 },
              { env_id: 1, run_index: 0, success: false, score: 0.0, duration: 5 }],
        };
        console.log(JSON.stringify(joinTaskEpisodes(taskRow, runs, eps)));
    """))
    assert [r["policy"] for r in got] == ["pi05", "cosmos3", "cosmos3"]
    assert [r["run_id"] for r in got] == ["a", "b", "b"]
    assert got[0]["success"] is True and got[1]["success"] is False


@node
def test_a_run_with_no_episodes_contributes_nothing():
    """A task missing from an experiment must not break the page."""
    got = _run(textwrap.dedent("""
        console.log(JSON.stringify(joinTaskEpisodes(
          { runs: ['a', 'b'] }, [{ run_id: 'a', policy: 'p' }], { a: [], b: undefined })));
    """))
    assert got == []


@node
def test_unknown_run_still_yields_a_row():
    """An episode whose experiment is missing from /api/runs is labelled, not dropped."""
    got = _run(textwrap.dedent("""
        console.log(JSON.stringify(joinTaskEpisodes(
          { runs: ['ghost'] }, [], { ghost: [{ env_id: 0, run_index: 0, success: true }] })));
    """))
    assert len(got) == 1 and got[0]["policy"] == "unknown"
    assert got[0]["score"] is None and got[0]["duration"] is None


@node
def test_policy_summary_counts_runs_not_episodes():
    got = _run(textwrap.dedent("""
        const rows = [
          { policy: 'pi05', run_id: 'a', success: true,  score: 1.0 },
          { policy: 'pi05', run_id: 'a', success: false, score: 0.0 },
          { policy: 'pi05', run_id: 'b', success: true,  score: 0.5 },
          { policy: 'gem',  run_id: 'c', success: false, score: null },
        ];
        console.log(JSON.stringify(summarisePolicies(rows)));
    """))
    pi05 = next(p for p in got if p["policy"] == "pi05")
    assert pi05["n"] == 3 and pi05["s"] == 2 and pi05["runs"] == 2
    assert abs(pi05["rate"] - 2 / 3) < 1e-9
    assert abs(pi05["score"] - 0.5) < 1e-9          # mean of 1.0, 0.0, 0.5
    gem = next(p for p in got if p["policy"] == "gem")
    assert gem["score"] is None, "no scored episodes must give None, not 0"
    assert [p["policy"] for p in got] == ["pi05", "gem"], "busiest policy first"


def test_the_overview_task_cell_is_a_link_and_the_route_exists():
    src = APP_JS.read_text()
    assert "onclick: () => selectTaskAll(t.task)" in src, "overview task names must be clickable"
    assert "if (parts[1] === 'task' && parts[2]) { await selectTaskAll(parts[2]); return; }" in src, \
        "#/results/task/<Task> must resolve so the page is linkable (H-E15)"
    assert "state.overview = null;" in src, "the overview cache must be cleared in refreshAll"
