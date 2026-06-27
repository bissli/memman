"""Cron-expression parsing and native-scheduler translation.

memman's scheduler natively models fixed integer-second intervals.
The backup feature instead accepts a 5-field cron string and renders
it to each platform's native calendar scheduler at install time:
systemd `OnCalendar=`, launchd `StartCalendarInterval`, and an
in-process matcher (`cron_matches`) for serve mode.

Fields are `minute hour day-of-month month day-of-week`. Per field:
`*`, an int, a list `a,b`, a range `a-b`, a step `*/n`, or a
range-step `a-b/n`. Day-of-week `0` and `7` both mean Sunday. Times
are interpreted in LOCAL time (cron convention; matches the
systemd/launchd defaults). Bundle filenames use UTC for stable sort.
"""

from datetime import datetime

_FIELD_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 59),
    (0, 23),
    (1, 31),
    (1, 12),
    (0, 7),
    )

_DOW_ORDER: tuple[tuple[int, str], ...] = (
    (1, 'Mon'),
    (2, 'Tue'),
    (3, 'Wed'),
    (4, 'Thu'),
    (5, 'Fri'),
    (6, 'Sat'),
    (0, 'Sun'),
    )


def _parse(expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    """Split `expr` into 5 expanded sets; normalize dow `7` to `0`.

    Raises ValueError when the field count is not exactly 5 or any
    field fails to expand (out-of-range value, inverted range,
    non-positive step, or non-integer token).
    """
    def expand(spec: str, lo: int, hi: int) -> set[int]:
        result: set[int] = set()
        for part in spec.split(','):
            part = part.strip()
            if not part:
                raise ValueError(f'empty field part in {spec!r}')
            step = 1
            base = part
            if '/' in part:
                base, _, step_str = part.partition('/')
                try:
                    step = int(step_str)
                except ValueError:
                    raise ValueError(f'invalid step {step_str!r} in {spec!r}')
                if step <= 0:
                    raise ValueError(f'step must be positive in {spec!r}')
            if base == '*':
                start, end = lo, hi
            elif '-' in base:
                start_str, _, end_str = base.partition('-')
                try:
                    start, end = int(start_str), int(end_str)
                except ValueError:
                    raise ValueError(f'invalid range {base!r} in {spec!r}')
            else:
                try:
                    start = end = int(base)
                except ValueError:
                    raise ValueError(f'invalid value {base!r} in {spec!r}')
            if start < lo or end > hi or start > end:
                raise ValueError(
                    f'value out of range in {spec!r} (bounds {lo}-{hi})')
            result.update(range(start, end + 1, step))
        return result

    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(
            f'cron expr must have 5 fields, got {len(fields)}: {expr!r}')
    minute, hour, dom, month, dow = (
        expand(spec, lo, hi)
        for spec, (lo, hi) in zip(fields, _FIELD_BOUNDS))
    if 7 in dow:
        dow = (dow - {7}) | {0}
    return minute, hour, dom, month, dow


def cron_matches(expr: str, dt: datetime) -> bool:
    """Return True iff local-time `dt` matches the 5-field cron `expr`.

    Applies the Vixie day-of-month / day-of-week rule: when BOTH dom
    and dow are restricted (neither is `*`), a day matches on EITHER;
    otherwise the restricted field alone decides. `dt.weekday()`
    (Mon=0) maps to cron dow (Sun=0) via `(weekday + 1) % 7`.
    """
    minute, hour, dom, month, dow = _parse(expr)
    fields = expr.split()
    dom_is_star = fields[2].strip() == '*'
    dow_is_star = fields[4].strip() == '*'

    if dt.minute not in minute:
        return False
    if dt.hour not in hour:
        return False
    if dt.month not in month:
        return False

    cron_dow = (dt.weekday() + 1) % 7
    dom_match = dt.day in dom
    dow_match = cron_dow in dow
    if dom_is_star and dow_is_star:
        return True
    if dom_is_star:
        return dow_match
    if dow_is_star:
        return dom_match
    return dom_match or dow_match


def _oncal_token(raw: str, values: set[int]) -> str:
    """Render one OnCalendar component: `*`, a zero-padded single, or a list.

    A literal `*` field passes through. A single value is zero-padded
    to width 2; a multi-value field is an unpadded comma list.
    """
    if raw.strip() == '*':
        return '*'
    ordered = sorted(values)
    if len(ordered) == 1:
        return f'{ordered[0]:02d}'
    return ','.join(str(v) for v in ordered)


def cron_to_oncalendar(expr: str) -> str:
    """Render a systemd `OnCalendar=` value from a cron expr.

    Emits `<DOW> <Y-M-D> <H:M:S>` with the year always `*` and no
    `UTC` suffix (so systemd interprets it in local time). The DOW
    token is omitted when day-of-week is `*`. When BOTH day-of-month
    and day-of-week are restricted, systemd evaluates them as AND
    (not the cron OR); `cron_matches` is the OR-correct path for
    serve mode.
    """
    minute, hour, dom, month, dow = _parse(expr)
    fields = expr.split()
    date = f'*-{_oncal_token(fields[3], month)}-{_oncal_token(fields[2], dom)}'
    time = (f'{_oncal_token(fields[1], hour)}'
            f':{_oncal_token(fields[0], minute)}:00')
    if fields[4].strip() == '*':
        return f'{date} {time}'
    dow_names = ','.join(
        name for value, name in _DOW_ORDER if value in dow)
    return f'{dow_names} {date} {time}'


def cron_to_launchd(expr: str) -> dict[str, int] | list[dict[str, int]]:
    """Render a launchd `StartCalendarInterval` value from a cron expr.

    Returns a single dict when every restricted field has one value,
    or a list of dicts (the cartesian product) when any restricted
    field has multiple values. `*` fields are omitted. When BOTH
    day-of-month and day-of-week are restricted, a Day-keyed group and
    a Weekday-keyed group are concatenated to preserve the cron OR
    semantics. launchd `Weekday` uses 0=Sunday.
    """
    minute, hour, dom, month, dow = _parse(expr)
    fields = [f.strip() for f in expr.split()]

    base: list[tuple[str, list[int]]] = []
    if fields[0] != '*':
        base.append(('Minute', sorted(minute)))
    if fields[1] != '*':
        base.append(('Hour', sorted(hour)))
    if fields[3] != '*':
        base.append(('Month', sorted(month)))

    day_specs: list[tuple[str, list[int]]] = []
    if fields[2] != '*':
        day_specs.append(('Day', sorted(dom)))
    if fields[4] != '*':
        day_specs.append(('Weekday', sorted(dow)))

    if len(day_specs) == 2:
        dicts = (_product(base + [day_specs[0]])
                 + _product(base + [day_specs[1]]))
    elif len(day_specs) == 1:
        dicts = _product(base + day_specs)
    else:
        dicts = _product(base)

    return dicts[0] if len(dicts) == 1 else dicts


def _product(field_pairs: list[tuple[str, list[int]]]) -> list[dict[str, int]]:
    """Cartesian product of `(key, values)` pairs into a list of dicts."""
    dicts: list[dict[str, int]] = [{}]
    for key, values in field_pairs:
        expanded: list[dict[str, int]] = []
        for partial in dicts:
            for value in values:
                merged = dict(partial)
                merged[key] = value
                expanded.append(merged)
        dicts = expanded
    return dicts
