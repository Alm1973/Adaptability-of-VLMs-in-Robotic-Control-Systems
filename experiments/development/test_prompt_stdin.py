import sys
import threading
import time

import final_experiment as fx

LINES = ["", "y", "", "r", "", "s", "", "i", "", "a"]
SEQS = [("go", ["", "s", "a"]), ("ok?", ["", "y"]), ("go", ["", "s", "a"]),
        ("acc", ["", "r", "i"]), ("go", ["", "s", "a"]),
        ("act", ["", "s", "a"]), ("ok?", ["", "y"]),
        ("acc", ["", "r", "i"]), ("ok?", ["", "y"]), ("go", ["", "s", "a"])]


class Keyboard:

    def __init__(self):
        self.gate = threading.Semaphore(0)
        self.lines = []
        self.lock = threading.Lock()

    def type(self, line):
        with self.lock:
            self.lines.append(line)
        self.gate.release()

    def readline(self):
        self.gate.acquire()
        with self.lock:
            return self.lines.pop(0) + "\n"

    def isatty(self):
        return False


kb = Keyboard()
sys.stdin = kb

got = []


def run():
    for msg, allowed in SEQS:
        got.append(fx.prompt(msg, allowed))


t = threading.Thread(target=run, daemon=True)
t.start()

for line in LINES:
    time.sleep(0.25)
    kb.type(line)

t.join(timeout=20)

print()
if t.is_alive():
    print(f"FAIL: hung after {len(got)} of {len(SEQS)} prompts -> {got}")
    print("      This is the orphaned-reader bug. Do NOT run an experiment.")
    sys.exit(1)

print(f"consumed {len(got)}/{len(SEQS)} prompts: {got}")
if got != LINES:
    print(f"FAIL: wrong lines\n  got  {got}\n  want {LINES}")
    sys.exit(1)
print("PASS: every prompt received exactly its own line, in order.")
print("PASS: no prompt hung -- one reader thread, one queue.")
