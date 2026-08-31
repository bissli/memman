"""Store names Postgres cannot host must not crash `memman migrate`.

`valid_store_name` accepts `[a-zA-Z0-9][a-zA-Z0-9_-]*`, so stores
already on disk carry hyphens or a leading digit. `_check_identifier`
is narrower (`[a-zA-Z_][a-zA-Z0-9_]*`), and `list_local_store_dirs`
scans the filesystem without consulting either rule, so migrate meets
those names whatever the creation path allows. Both directions of the
migration touch a Postgres schema, so both must skip such names under
`--all` and refuse them under `--store NAME`, never traceback.
"""

import sqlite3
from pathlib import Path

import pytest
from memman.store.db import _BASELINE_SCHEMA
from memman.store.errors import ConfigError
from tests.conftest import _set_env_file_value, invoke

FAKE_DSN = 'postgresql://u:p@h:5432/d'


@pytest.fixture
def runner(mm_runner):
    return mm_runner


def _seed_store(data_dir, name, dim=512):
    """Create a migratable SQLite store carrying one insight."""
    sdir = Path(data_dir) / 'data' / name
    sdir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sdir / 'memman.db'))
    try:
        conn.executescript(_BASELINE_SCHEMA)
        conn.execute(
            'insert into meta (key, value) values (?, ?)',
            ('embed_fingerprint',
             ('{"provider":"voyage","model":"voyage-3-lite","dim":'
              f'{dim}}}')))
        conn.execute(
            'insert into insights (id, content, category, importance,'
            ' entities, source, created_at, updated_at)'
            ' values (?, ?, ?, ?, ?, ?, ?, ?)',
            ('11111111-1111-4111-8111-111111111111', 'seed text',
             'fact', 3, '[]', 'user', '2026-01-01T00:00:00Z',
             '2026-01-01T00:00:00Z'))
        conn.commit()
    finally:
        conn.close()


def _stub_postgres(monkeypatch, *, preflight=None):
    """Seed a default DSN and stub the two Postgres round-trips.

    Without the DSN the command exits at its own missing-DSN gate,
    which would let these tests pass while the crash is untouched.
    """
    import memman.migrate as mig

    _set_env_file_value('MEMMAN_DEFAULT_POSTGRES_DSN', FAKE_DSN)
    monkeypatch.setattr(
        mig, 'preflight', preflight or (lambda dsn: {'select_1': True}))
    monkeypatch.setattr(
        mig, 'inspect_target_schemas',
        lambda dsn, stores: dict.fromkeys(stores, mig.SchemaState.ABSENT))


def test_migrate_named_store_rejects_unhostable_name(runner, monkeypatch):
    """`--store demo-v3` fails cleanly instead of raising ConfigError.

    Mutation: dropping the eligibility guard, so the command skips
    the unhostable store and exits 0 instead of refusing.
    Oracle: the non-zero exit. The CLI root group also catches
    ConfigError, so the escaped-type assertion below holds either
    way; the exit code is what separates a refusal from a skip.
    """
    _, data_dir = runner
    _seed_store(data_dir, 'demo-v3')
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--store', 'demo-v3', '--to', 'postgres',
                 '--dry-run'])

    assert not isinstance(result.exception, ConfigError), result.output
    assert result.exit_code != 0
    assert 'demo-v3' in result.output


def test_migrate_refusal_names_the_remedy(runner, monkeypatch):
    """The refusal tells the operator where the store can stay.

    Mutation: a bare `except ConfigError: raise ClickException(str(e))`
    echoing only "invalid SQL identifier", naming neither the store
    nor a remedy.
    Oracle: the message must carry the offending store AND the word
    sqlite, the backend it can remain on.
    """
    _, data_dir = runner
    _seed_store(data_dir, 'demo-v3')
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--store', 'demo-v3', '--to', 'postgres',
                 '--dry-run'])

    assert 'demo-v3' in result.output
    assert 'sqlite' in result.output.lower()


def test_migrate_named_store_refuses_before_connecting(runner,
                                                       monkeypatch):
    """An unhostable name is rejected without a Postgres round-trip.

    Mutation: placing the guard after `preflight(dsn)`, making a pure
    naming error depend on a reachable server.
    Oracle: a preflight stub that fails the test if it is ever called.
    """
    def _boom(dsn):
        raise AssertionError('preflight must not run for a bad name')

    _, data_dir = runner
    _seed_store(data_dir, 'demo-v3')
    _stub_postgres(monkeypatch, preflight=_boom)

    result = invoke(
        runner, ['migrate', '--store', 'demo-v3', '--to', 'postgres',
                 '--dry-run'])

    assert not isinstance(result.exception, AssertionError), result.output
    assert result.exit_code != 0
    assert 'demo-v3' in result.output


def test_migrate_all_skips_unhostable_and_keeps_the_rest(runner,
                                                         monkeypatch):
    """`--all` migrates the valid stores and skips the unhostable one.

    Mutation: aborting the whole run on the first bad name, so every
    valid store stays behind.
    Oracle: the dry-run line for the good store must be present while
    no `store_demo-v3` destination is ever proposed.
    """
    _, data_dir = runner
    _seed_store(data_dir, 'demo-v3')
    _seed_store(data_dir, 'goodstore')
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--all', '--to', 'postgres', '--dry-run'])

    assert result.exit_code == 0, result.output
    assert 'goodstore: insights=1' in result.output
    assert 'store_demo-v3' not in result.output


def test_migrate_all_reports_the_skip(runner, monkeypatch):
    """The skipped store is named, not silently dropped.

    Mutation: filtering the bad name out of `todo` with no message,
    leaving the operator believing every store migrated.
    Oracle: the offending name appears alongside a skip word.
    """
    _, data_dir = runner
    _seed_store(data_dir, 'demo-v3')
    _seed_store(data_dir, 'goodstore')
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--all', '--to', 'postgres', '--dry-run'])

    assert 'demo-v3' in result.output
    assert 'skip' in result.output.lower()


def test_migrate_all_with_only_unhostable_stores_exits_clean(
        runner, monkeypatch):
    """Filtering every store reports a zero run, not an empty plan.

    Mutation: dropping the `if not todo` early return, which falls
    through and prints `Stores (0):` with no summary at all -- the
    skip echo alone still satisfies an exit-code-plus-name check, so
    the count line is the only assertion with teeth here.
    Oracle: the `planned=0 skipped=1` summary, and the absence of a
    migration plan header. A dry run never reports "migrated".
    """
    _, data_dir = runner
    _seed_store(data_dir, 'demo-v3')
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--all', '--to', 'postgres', '--dry-run'])

    assert result.exit_code == 0, result.output
    assert 'demo-v3' in result.output
    assert 'planned=0 skipped=1' in result.output
    assert 'Migration plan' not in result.output


def test_leading_digit_store_name_is_also_unhostable(runner, monkeypatch):
    """A leading-digit name is refused for the same reason.

    Postgres would in fact accept the schema `store_9lives`; it is
    `_store_schema` checking the unprefixed name that refuses. The
    guard must follow that check, not a rule of its own.

    Mutation: guarding on the hyphen alone (`'-' in name`), which lets
    `9lives` through to the same ConfigError crash.
    Oracle: `valid_store_name('9lives')` is True, so a guard keyed on
    anything but `_store_schema` lets it through.
    """
    from memman.store.db import valid_store_name

    assert valid_store_name('9lives')

    _, data_dir = runner
    _seed_store(data_dir, '9lives')
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--store', '9lives', '--to', 'postgres',
                 '--dry-run'])

    assert not isinstance(result.exception, ConfigError), result.output
    assert result.exit_code != 0
    assert '9lives' in result.output


def test_existing_hyphenated_store_stays_readable(runner):
    """A hyphenated store stays fully usable on sqlite.

    Mutation: narrowing `valid_store_name` to the Postgres identifier
    rule to stop such names arising, which would strand every
    hyphenated store already on disk -- there is no `store rename`,
    and `_validate_namespace` (store/config.py) would then reject a
    `MEMMAN_POSTGRES_DSN_<store>` row outright.
    Oracle: `status` on the hyphenated store still reports it.
    """
    _, data_dir = runner
    _seed_store(data_dir, 'demo-v3')

    result = invoke(runner, ['--store', 'demo-v3', 'status'])

    assert result.exit_code == 0, result.output
    assert 'demo-v3' in result.output


def test_inspect_target_schemas_raises_before_it_connects():
    """Pin the crash site the CLI guard exists to keep unreachable.

    `inspect_target_schemas` builds its schema map before opening a
    connection, so an unhostable name raises ConfigError against an
    unroutable DSN. The CLI stubs this call in the tests above, so
    without this case nothing would hold the raw hazard in place.

    Mutation: relaxing `_check_identifier` to accept a hyphen, which
    would silently emit the unquoted DDL `create schema store_demo-v3`.
    Oracle: ConfigError, raised against a DSN pointing at a closed
    port -- a connection attempt would surface OperationalError.
    """
    from memman.migrate import inspect_target_schemas

    with pytest.raises(ConfigError):
        inspect_target_schemas(
            'postgresql://nobody@127.0.0.1:1/nodb', ['demo-v3'])


def test_migrate_to_sqlite_refuses_unhostable_name(runner, monkeypatch):
    """The reverse direction refuses the name too, and leaks nothing.

    A postgres route can be hand written into the env file, so
    `--to sqlite` meets the same names. It reaches `preflight_source`,
    which calls `_check_identifier` on the raw store name.

    Mutation: scoping the guard to `if target_backend == 'postgres'`.
    The widened scratch-cleanup handler still turns the ConfigError
    into a clean exit, so an exception-type check cannot see this;
    what changes is that the refusal then lands AFTER the plan prints
    and the drain lock is taken, carrying the bare text "invalid SQL
    identifier" instead of a remedy.
    Oracle: no migration plan is printed at all, and the message
    carries the portable suggestion `demo_v3`.
    """
    from tests.conftest import _set_env_file_value

    _, data_dir = runner
    _set_env_file_value('MEMMAN_BACKEND_demo-v3', 'postgres')
    _set_env_file_value('MEMMAN_POSTGRES_DSN_demo-v3', FAKE_DSN)

    result = invoke(
        runner, ['migrate', '--store', 'demo-v3', '--to', 'sqlite',
                 '--yes'])

    assert not isinstance(result.exception, ConfigError), result.output
    assert result.exit_code != 0
    assert 'demo-v3' in result.output
    assert 'Migration plan' not in result.output
    assert 'demo_v3' in result.output
    leaked = [p.name for p in Path(data_dir).iterdir()
              if p.name.startswith('migrate-')]
    assert leaked == [], leaked


def test_portable_suggestion_is_always_creatable():
    """Every suggestion clears both name rules, not just the SQL one.

    Mutation: rewriting only the hyphen (the first implementation),
    which returned `default.bak` unchanged -- telling the operator to
    create a name `memman store create` rejects.
    Oracle: `_check_identifier` and `valid_store_name` both accept
    the output for every input, including a dotted directory name.
    """
    from memman.store.backend import _check_identifier
    from memman.store.db import portable_store_name, valid_store_name

    for name in ['demo-v3', '9lives', 'default.bak', 'demo v3',
                 '_x', 'x.y-z 1', '']:
        suggestion = portable_store_name(name)
        _check_identifier(suggestion)
        assert valid_store_name(suggestion), (name, suggestion)


def test_migrate_refuses_a_store_that_does_not_exist(runner, monkeypatch):
    """An unknown store is reported as unknown, not as sqlite-backed.

    Mutation: taking `--store NAME` on trust, so the naming guard
    describes a store that is not there ("stays fully usable on
    sqlite") instead of saying it does not exist.
    Oracle: the message must say the store does not exist and must
    not claim it remains usable.
    """
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--store', 'nope-1', '--to', 'postgres',
                 '--dry-run'])

    assert result.exit_code != 0
    assert 'nope-1' in result.output
    assert 'does not exist' in result.output.lower()
    assert 'stays fully usable' not in result.output


def test_suggestion_that_names_a_live_store_says_so(runner, monkeypatch):
    """A colliding suggestion warns instead of pointing at real data.

    `portable_store_name('a-b')` is `a_b`. When `a_b` already exists,
    telling the operator to "create a_b and migrate that" points at a
    different store holding unrelated rows.

    Mutation: emitting the suggestion unconditionally, with no check
    that it names an existing store.
    Oracle: with both `a-b` and `a_b` present the output must flag
    the collision; the plain suggestion wording must not appear.
    """
    _, data_dir = runner
    _seed_store(data_dir, 'a-b')
    _seed_store(data_dir, 'a_b')
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--all', '--to', 'postgres', '--dry-run'])

    assert 'a_b' in result.output
    assert 'already' in result.output.lower()
    assert 'taken' in result.output.lower()


def test_dry_run_reports_the_skip_count_alongside_a_plan(runner,
                                                         monkeypatch):
    """A mixed dry run summarizes skips like every other branch.

    Mutation: returning from the dry-run branch before the summary,
    so a run that skipped stores looks like a clean full plan.
    Oracle: the `skipped=1` count appears even though one store was
    planned.
    """
    _, data_dir = runner
    _seed_store(data_dir, 'demo-v3')
    _seed_store(data_dir, 'goodstore')
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--all', '--to', 'postgres', '--dry-run'])

    assert result.exit_code == 0, result.output
    assert 'goodstore: insights=1' in result.output
    assert 'skipped=1' in result.output


def test_refusal_quotes_the_actual_reason_not_a_guess(runner, monkeypatch):
    """A too-long name is refused for length, not for its characters.

    `_store_schema` rejects on character class AND on length. Asserting
    a reason instead of quoting the raised one told the owner of a
    60-character name that it failed a pattern it matches, and offered
    the identical name as the remedy.

    Mutation: hardcoding the refusal text rather than passing through
    the `ConfigError` message.
    Oracle: the message must say the name is too long and must not
    claim the identifier pattern was violated.
    """
    _, data_dir = runner
    name = 'a' * 60
    _seed_store(data_dir, name)
    _stub_postgres(monkeypatch)

    result = invoke(
        runner, ['migrate', '--store', name, '--to', 'postgres',
                 '--dry-run'])

    assert result.exit_code != 0
    assert 'too long' in result.output
    assert '[a-zA-Z_][a-zA-Z0-9_]*' not in result.output
    assert f"create '{name}'" not in result.output
