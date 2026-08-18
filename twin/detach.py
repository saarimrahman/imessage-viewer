#!/usr/bin/env python3
"""Start a command in its own session so it survives the parent shell.

macOS has no `setsid`, and a job started with `nohup ... &` still shares the
process group of the shell that started it. When that shell is killed, the job
dies with it. This launcher calls `os.setsid` between the fork and the exec, so
the child keeps running.

Usage: detach.py LOGFILE COMMAND [ARG ...]
"""

import os
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: detach.py LOGFILE COMMAND [ARG ...]")
    log_path = sys.argv[1]
    command = sys.argv[2:]

    pid = os.fork()
    if pid > 0:
        print(pid)
        return

    os.setsid()
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
