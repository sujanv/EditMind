#!/usr/bin/env python3
"""
Staggered Git Pusher for EditMind.

Pushes local commits to the remote repository sequentially one at a time,
introducing a random delay between 30 and 45 minutes (1800 - 2700 seconds)
between successive pushes to simulate continuous gradual development.
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys
import time

STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".pusher_state.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "staggered_pusher.log")


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def run_cmd(cmd: list) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {res.stderr.strip()}")
    return res.stdout.strip()


def get_unpushed_commits(remote_branch: str = "origin/main", local_branch: str = "main") -> list:
    """Returns list of commit hashes and subjects ahead of remote, oldest first."""
    try:
        # Check if remote ref exists
        out = run_cmd(["git", "log", f"{remote_branch}..{local_branch}", "--reverse", "--format=%H %s"])
        if not out:
            return []
        commits = []
        for line in out.splitlines():
            if line.strip():
                parts = line.strip().split(" ", 1)
                commits.append({"hash": parts[0], "subject": parts[1] if len(parts) > 1 else ""})
        return commits
    except Exception as e:
        log(f"Warning checking unpushed commits: {e}")
        return []


def push_single_commit(commit_hash: str, branch: str = "main", remote: str = "origin") -> bool:
    """Pushes up to commit_hash to remote branch."""
    refspec = f"{commit_hash}:refs/heads/{branch}"
    log(f"Pushing commit {commit_hash[:8]} to {remote}/{branch}...")
    try:
        subprocess.run(["git", "push", remote, refspec], check=True)
        return True
    except subprocess.CalledProcessError as e:
        log(f"Git push failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="EditMind Staggered Git Pusher")
    parser.add_argument("--min-delay", type=int, default=1800, help="Minimum delay in seconds (default: 1800s / 30m)")
    parser.add_argument("--max-delay", type=int, default=2700, help="Maximum delay in seconds (default: 2700s / 45m)")
    parser.add_argument("--push-first-now", action="store_true", default=False, help="Push the first commit immediately without initial delay")
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument("--remote", type=str, default="origin")
    args = parser.parse_args()

    log("EditMind Staggered Pusher Daemon initialized.")
    log(f"Interval configured: {args.min_delay // 60}m - {args.max_delay // 60}m random jitter.")

    commits = get_unpushed_commits(f"{args.remote}/{args.branch}", args.branch)
    if not commits:
        log("No unpushed commits found. Working tree up to date with remote.")
        return

    log(f"Found {len(commits)} commits queued for sequential push:")
    for idx, c in enumerate(commits):
        log(f"  [{idx+1}/{len(commits)}] {c['hash'][:8]} - {c['subject']}")

    for idx, c in enumerate(commits):
        if idx == 0 and args.push_first_now:
            log(f"Pushing commit 1/{len(commits)} immediately without delay...")
        else:
            delay = random.randint(args.min_delay, args.max_delay)
            next_time = (datetime.datetime.now() + datetime.timedelta(seconds=delay)).strftime("%H:%M:%S")
            log(f"Waiting {delay // 60} minutes {delay % 60} seconds (until ~{next_time}) before next push...")
            time.sleep(delay)

        success = push_single_commit(c["hash"], branch=args.branch, remote=args.remote)
        if success:
            log(f"Successfully pushed commit [{idx+1}/{len(commits)}] {c['hash'][:8]}: '{c['subject']}'")
        else:
            log(f"Failed to push {c['hash'][:8]}. Retrying in 60 seconds...")
            time.sleep(60)
            push_single_commit(c["hash"], branch=args.branch, remote=args.remote)

    log("All queued commits pushed successfully! Daemon exiting.")


if __name__ == "__main__":
    main()
