---
name: code-review
description: Reviews a diff before it is committed or merged - correctness, numerics, performance, structure, naming, security and tests. Use before every commit, after finishing a feature or fix, and whenever asked to review code written by you, another agent or a person.
---

# Code Review

Review your own diff the way a skeptical maintainer of a large open-source project would review a stranger's pull request. AI-written code is fluent and confident even when it is wrong, so it gets more scrutiny, not less.

## Process

1. **Understand the intent.** What was the task? What behavior should change, and what must not?
2. **Read the tests first.** They show what the author thinks the code does. Are the edge cases, dtypes and error paths there?
3. **Read the diff** (`git diff --staged`) file by file against the axes below.
4. **Run the checks**: `pre-commit run -a`, `pytest -m "not gpu"`, and `pytest -m gpu` plus the relevant benchmark if tensor code changed.
5. **Try to break it.** Invert one condition the change added, or perturb a kernel's output, and rerun the tests. If they stay green, a test is missing.
6. **Write the findings**, highest leverage first, then fix them before committing.

## Axes, in order of importance

### 1. Correctness
- Does it do what the task asked, and nothing else?
- Empty batch, batch of one, maximum length, non-power-of-two sizes?
- Every error path handled, and none swallowed?
- Off-by-one errors in token positions, block indices, slicing and padding?
- State that can go stale: caches, preallocated buffers reused across steps, sequences that finished or were preempted?

### 2. Numerics
- Precision changes explicit? Reductions accumulated in fp32?
- Masks, `-inf` and NaN handled? What happens on an all-masked row?
- Compared against a reference with per-dtype tolerances, and were any tolerances loosened?

### 3. Performance
- Any host-device sync in the hot path (`.item()`, `.cpu()`, `.tolist()`, tensor-valued `if`, `print`)?
- Python loops over tokens, heads or batch elements?
- Per-step allocations, hidden copies (`.contiguous()`, `cat`, advanced indexing) in loops?
- Breaks CUDA graph capture or adds `torch.compile` graph breaks?
- Is there a benchmark for any performance claim, with the setup recorded? Does the improvement exceed the noise?

### 4. Structure
- Does logic live in the right module (engine vs. server, kernel vs. reference)?
- Does it reuse the existing helper instead of adding a near-duplicate?
- Is a new conditional bolted onto an unrelated flow? That signals a missing abstraction.
- Does a refactor reduce the number of concepts a reader must hold, or just move them?
- Did a file grow past ~500 lines? Split before adding more.
- Is any abstraction used exactly once? Inline it.

### 5. Naming and readability
- Run the `naming` checklist on every new or renamed identifier, and check that new tensors have shape comments.
- Any comment that explains *what*? Rename or extract instead.
- Any nested comprehension, `reduce` or chained conditional expression outside a tensor hot path? Rewrite as an explicit loop (see `production-code`).
- Leftovers: commented-out code, debug prints, unused imports, `TODO`s without an issue.

### 6. Security and robustness
- Untrusted input (HTTP bodies, config files, model configs, `trust_remote_code`) parsed at the edge?
- No `pickle.load` or `torch.load(weights_only=False)` on files you don't control?
- No secrets or tokens in code, logs or fixtures?
- Queues and concurrency bounded?

### 7. Tests
- Does each changed behavior have a test that fails without the change?
- Does a bug fix include the reproducing test?
- Are GPU tests marked, and do CPU-only tests still pass without CUDA?

## Writing findings

Label every finding so it's clear what must change:

| Label | Meaning |
| --- | --- |
| **Critical** | Blocks the commit: wrong output, crash, data or memory corruption, security hole |
| *(no label)* | Required: fix before committing |
| **Consider** | A real improvement, but optional |
| **Nit** | Style or taste; fine to ignore |

Each finding names the file and line, states the problem in one sentence, and proposes the fix. "This could be slow" is not a finding; "`seq_lens.tolist()` on line 88 syncs the GPU every decode step; keep a CPU copy updated by the scheduler instead" is.

Lead with what matters. One correctness bug outweighs ten nits.

## Be honest

- Don't approve because the tests pass. Tests don't catch syncs, bad structure or bad names.
- Don't soften a real bug into "might be worth a look".
- Quantify: "This allocates a [num_seqs, vocab_size] fp32 tensor per step, about 600 MB at batch 1,024 with a 150k vocabulary."
- If the whole approach is wrong, say so and propose the alternative.

## Dead code

List any code the change made unused and remove it. If you find pre-existing dead code outside the change, list it and ask before deleting it.

## Verdict

End with one of:
- **Ready to commit**: all checks pass, no Critical or Required findings left.
- **Changes needed**: list what remains.
