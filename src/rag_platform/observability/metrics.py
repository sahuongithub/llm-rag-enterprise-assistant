from __future__ import annotations

from collections import defaultdict
from time import perf_counter


class CounterRegistry:
    def __init__(self) -> None:
        self._counters: defaultdict[str, int] = defaultdict(int)
        self.started_at = perf_counter()

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] += value

    def snapshot(self) -> dict[str, int]:
        return dict(self._counters)

    def prometheus(self, gauges: dict[str, int | float | str] | None = None) -> str:
        lines: list[str] = []
        for name, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        for name, value in sorted((gauges or {}).items()):
            if isinstance(value, str):
                lines.append(f'{name}{{value="{value}"}} 1')
            else:
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
        lines.append("# TYPE process_uptime_seconds gauge")
        lines.append(f"process_uptime_seconds {round(perf_counter() - self.started_at, 3)}")
        return "\n".join(lines) + "\n"
