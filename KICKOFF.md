# Kickoff: running this project with Claude Code on glogin

## 1. One-time setup on glogin

```bash
ssh <your-cs-username>@glogin.cs.vt.edu

# Install Claude Code into your home directory (no sudo needed)
curl -fsSL https://claude.ai/install.sh | bash
exec $SHELL -l
claude --version

# Create the repo and unpack the starter kit into it
mkdir -p ~/personal_projects/decode-sol-gb10
cd ~/personal_projects/decode-sol-gb10
git init
unzip ~/decode-sol-gb10-kit.zip -d .    # after copying the zip over with scp
git add -A
git commit -m "[Docs] Add project brief, agent instructions and skills"
```

The first time you run `claude`, it prints a login link: open it in your laptop's browser and paste the code back into the terminal.

Run Claude Code inside `tmux` (`tmux new -s sol`), so a dropped SSH connection doesn't kill the session. Reattach with `tmux attach -t sol`. The session can stay open while you're away, but GPU runs should only happen while you're watching.

Don't run Claude Code with `--dangerously-skip-permissions` on this machine. The included `.claude/settings.json` blocks `sudo`, containers and clock changes, and still asks you before anything else.

## 2. Kickoff prompt (paste into Claude Code)

```
Read CLAUDE.md and docs/PROJECT_BRIEF.md in full, then the skills listed in CLAUDE.md.

We are starting milestone M0 (environment). Before running anything:
1. Summarize the project goal and the shared-machine rules back to me in a few lines.
2. List anything in the brief that is unclear or that you think is wrong.
3. Propose the exact commands you will run for M0, in order.

Then wait for my go-ahead. Work one milestone at a time, and stop at every checkpoint
in the brief. Never start a GPU process without telling me first.
```

## 3. Prompts for each later milestone

Use these after you've approved the previous checkpoint.

**M1**
```
M0 is approved. Start M1. Write the tests for roofline.py first, using hand-computed
examples, then implement it. Show me the bandwidth and idle-power results at the checkpoint.
```

**M2**
```
M1 is approved. Start M2: one grid point end to end, with energy. Before running,
show me the predicted step time and tokens/s for that point from roofline.py.
```

**M3**
```
M2 is approved. Print the full feasible grid and the GPU-minute estimate, then stop.
```
Then, after you've reviewed the grid and checked `nvidia-smi` yourself:
```
Run the sweep for <model> only. Report when it finishes.
```

**M4**
```
The sweep is done. Start M4. Show me the plots and your five most interesting findings,
each with the result files that support it.
```

**M5**
```
Start M5. Draft docs/REPORT.md and the vLLM issue text. Mark every place where you are
unsure or where I should add my own interpretation.
```

## 4. Reviewing the agent's work

At every checkpoint:
- Ask it to run the `code-review` skill on the diff, then read the diff yourself.
- Check the numbers make physical sense. Measured speed can never beat the 273 GB/s ceiling, and energy per token should fall as batch size rises.
- Commit.
