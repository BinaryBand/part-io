import subprocess
import sys


def run(*args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
    return r.returncode, r.stdout, r.stderr


cmd = sys.argv[1]
if cmd == "diff-staged":
    code, out, _ = run("git", "diff", "--staged")
    print(out, end="")
    sys.exit(code)
elif cmd == "status":
    code, out, _ = run("git", "status")
    print(out, end="")
    sys.exit(code)
elif cmd == "add":
    for f in sys.argv[2:]:
        code, out, err = run("git", "add", f)
        if code != 0:
            print(err, end="", file=sys.stderr)
            sys.exit(code)
elif cmd == "commit":
    code, out, err = run("git", "commit", "-m", sys.argv[2])
    print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)
    sys.exit(code)
else:
    print(f"unknown: {cmd}", file=sys.stderr)
    sys.exit(1)
