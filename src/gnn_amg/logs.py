"""Web summary and current rank-zero tail; stdin must already be tee'd in full."""
from collections import deque
from pathlib import Path
import re
import sys


def main():
    tail = Path(sys.argv[1])
    recent = deque(maxlen=1000)
    tag = re.compile(r"^\[\d+,(\d+)\]")
    for line in sys.stdin:
        match = tag.match(line)
        if match is None or int(match.group(1)) == 0:
            print(line, end="", flush=True)
        if match and int(match.group(1)) == 0:
            recent.append(line)
            tail.write_text("".join(recent))


if __name__ == "__main__":
    main()
