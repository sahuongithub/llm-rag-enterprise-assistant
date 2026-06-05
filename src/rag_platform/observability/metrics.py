from __future__ import annotations

from collections import defaultdict


class CounterRegistry:
    def __init__(self) -> None:
        self._counters: defaultdict[str, int] = defaultdict(int)

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] += value

    def snapshot(self) -> dict[str, int]:
        return dict(self._counters)

