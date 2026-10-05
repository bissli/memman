"""Exercise the installed Codex skill's shell contract and store boundaries."""

import json
import shlex
from pathlib import Path

import pytest
from memman import config
from memman import doctor as doctor_mod
from memman.cli import cli
from memman.queue import queue_db
from memman.setup.codex import install_codex
from memman.store.db import store_dir
from memman.store.sqlite import drop_sqlite_store
from tests.conftest import _create_seeded_store, force_drain, invoke


@pytest.mark.no_auto_drain
def test_installed_skill_examples_correct_a_queued_memory(mm_runner):
    """Verify the skill examples run and a correction hides the old row.

    Mutation: An example in SKILL.md names a wrong verb or flag or goes
    missing, replace drops replaced_id when queuing, or recall still returns
    the replaced memory.
    Oracle: Commands parsed from the installed SKILL.md; queue rows compared to
    hand-written tuples, then chain states "replaced" and "current".
    """
    skills = Path.home() / '.agents/skills'
    install_codex({'skills_dir': str(skills),
                   'config_dir': str(Path.home() / '.codex')})
    source = (skills / 'memman/SKILL.md').read_text()
    fill = {
        '<thought>': 'The billing retry cap stays at three.',
        '<new content>': 'The billing retry cap is now four.',
        '<query>': 'billing retry decisions',
        # Basic recall is a literal text match, so use a phrase in the
        # memory.
        '<keyword>': 'billing retry',
        }
    examples = {}
    for block in source.split('```bash\n')[1:]:
        for line in block.split('```', 1)[0].splitlines():
            template = line.split('#', 1)[0].strip()
            examples[template] = [fill.get(arg, arg)
                                  for arg in shlex.split(template)[1:]]
    recall = examples['memman recall "<query>"']

    initial = invoke(mm_runner, recall)
    assert initial.exit_code == 0, initial.output
    assert initial.stdout == ''
    remembered = invoke(
        mm_runner, examples['memman remember "<thought>"'])
    assert remembered.exit_code == 0, remembered.output
    old_id = json.loads(remembered.output)['id']
    corrected = invoke(mm_runner, [
        old_id if arg == '<id>' else arg
        for arg in examples['memman replace <id> "<new content>"']])
    assert corrected.exit_code == 0, corrected.output
    new_id = json.loads(corrected.output)['id']

    _, data_dir = mm_runner
    with queue_db(data_dir) as conn:
        queued = conn.execute(
            'select queue_uuid, replaced_id from queue order by id'
        ).fetchall()
    assert [tuple(row) for row in queued] == [
        (old_id, None), (new_id, old_id)]
    force_drain(data_dir)

    history = invoke(mm_runner, [
        old_id if arg == '<id>' else arg
        for arg in examples['memman insights show <id>']] + ['--history'])
    assert history.exit_code == 0, history.output
    assert [(row['id'], row['state'])
            for row in json.loads(history.output)['chain']] == [
                (old_id, 'replaced'), (new_id, 'current')]
    for args in (recall, examples['memman recall "<keyword>" --basic']):
        recalled = invoke(mm_runner, args)
        assert recalled.exit_code == 0, recalled.output
        assert new_id[:8] in recalled.output
        assert old_id[:8] not in recalled.output


@pytest.mark.parametrize(('selection', 'expected'), [
    ('configured', 'configured'), ('environment', 'environment'),
    ('flag', 'flag'),
])
def test_codex_documented_store_precedence_applies_to_reads_and_writes(
        monkeypatch, mm_runner, selection, expected):
    """Verify remember and recall both pick flag, then env, then active store.

    Mutation: _resolve_store_name reorders the three sources, or remember and
    recall resolve the store through different paths.
    Oracle: Three stores created by hand; the payload store name, that store's
    recall hit, and empty output from the other two stores.
    """
    for name in ('configured', 'environment', 'flag'):
        created = invoke(mm_runner, ['store', 'create', name])
        assert created.exit_code == 0, created.output
    selected = invoke(mm_runner, ['store', 'use', 'configured'])
    assert selected.exit_code == 0, selected.output
    if selection != 'configured':
        monkeypatch.setenv(config.STORE, 'environment')
    flags = ['--store', 'flag'] if selection == 'flag' else []
    saved = invoke(mm_runner, [*flags, 'remember',
                               'The quartz retry cap is four.'])
    assert saved.exit_code == 0, saved.output
    payload = json.loads(saved.output)
    assert payload['store'] == expected
    read = invoke(mm_runner, [*flags, 'recall', 'quartz', '--basic'])
    assert read.exit_code == 0, read.output
    assert payload['id'][:8] in read.output
    for other in {'configured', 'environment', 'flag'} - {expected}:
        read_other = invoke(mm_runner, ['--store', other, 'recall',
                                        'quartz', '--basic'])
        assert read_other.exit_code == 0, read_other.output
        assert read_other.output == ''


def test_data_dir_flag_overrides_environment_without_cross_store_writes(
        monkeypatch, tmp_path, mm_runner):
    """Verify --data-dir beats MEMMAN_DATA_DIR for both write and read.

    Mutation: The cli group prefers MEMMAN_DATA_DIR over --data-dir, so the
    write or the queue entry lands in the default dir.
    Oracle: The env-named dir must hold no queue.db and no default store dir;
    recall in the custom dir must find the saved id.
    """
    runner, original = mm_runner
    drop_sqlite_store('default', original)
    custom = tmp_path / 'custom installation'
    custom.mkdir()
    (custom / 'env').write_text((Path(original) / 'env').read_text())
    _create_seeded_store('default', str(custom))
    monkeypatch.setenv(config.DATA_DIR, original)
    saved = runner.invoke(cli, ['--data-dir', str(custom), 'remember',
                                'The quartz retry cap is four.'])
    assert saved.exit_code == 0, saved.output
    memory_id = json.loads(saved.output)['id']
    # CliRunner shares os.environ in-process; reset it to model a new shell.
    monkeypatch.setenv(config.DATA_DIR, original)
    read = runner.invoke(cli, ['--data-dir', str(custom), 'recall',
                               'quartz', '--basic'])
    assert read.exit_code == 0, read.output
    assert memory_id[:8] in read.output
    assert not (Path(original) / 'queue.db').exists()
    assert not Path(store_dir(original, 'default')).exists()


@pytest.mark.no_auto_drain
@pytest.mark.parametrize('selection', ['flag', 'environment', 'configured'])
def test_invalid_store_is_rejected_before_acknowledging_a_write(
        monkeypatch, mm_runner, selection):
    """Verify remember rejects an invalid store name before queueing.

    Mutation: _resolve_store_name skips valid_store_name for the flag, env, or
    active-file source, so a traversal name is queued and acknowledged.
    Oracle: A nonzero exit code, the error text, and a hand-written row count
    of 0 in queue.db.
    """
    flags = []
    if selection == 'flag':
        flags = ['--store', '../other']
    elif selection == 'environment':
        monkeypatch.setenv(config.STORE, '../other')
    else:
        (Path(mm_runner[1]) / 'active').write_text('../other\n')
    result = invoke(mm_runner, [*flags, 'remember',
                                'The quartz retry cap is four.'])
    assert result.exit_code != 0, result.output
    assert 'invalid store name' in result.output.lower()
    _, data_dir = mm_runner
    with queue_db(data_dir) as conn:
        assert conn.execute('select count(*) from queue').fetchone()[0] == 0


@pytest.mark.parametrize('absolute', [False, True], ids=['traversal', 'absolute'])
def test_recall_rejects_store_paths_without_creating_a_database(
        tmp_path, mm_runner, absolute):
    """Verify recall rejects traversal and absolute store names.

    Mutation: valid_store_name accepts a slash, or recall opens the store
    before validating, so a database appears at the path.
    Oracle: A path built by hand that must not exist afterward, plus a nonzero
    exit code and the error text.
    """
    _, data_dir = mm_runner
    outside = tmp_path / 'outside' if absolute else Path(data_dir) / 'other'
    store = str(outside) if absolute else '../other'
    result = invoke(mm_runner, ['--store', store, 'recall', 'quartz', '--basic'])
    assert result.exit_code != 0, result.output
    assert 'invalid store name' in result.output.lower()
    assert not outside.exists()


def test_store_use_cannot_select_a_name_runtime_rejects(mm_runner):
    r"""Verify store use refuses a name that other commands reject.

    Mutation: store_use drops its valid_store_name check, so a discovered
    legacy.name directory is written to the active file.
    Oracle: An active file holding "default\n" before the act and unchanged
    after it, plus the error text.
    """
    _, data_dir = mm_runner
    (Path(data_dir) / 'data/legacy.name').mkdir(parents=True)
    active = Path(data_dir) / 'active'
    active.write_text('default\n')
    result = invoke(mm_runner, ['store', 'use', 'legacy.name'])
    assert result.exit_code != 0, result.output
    assert 'invalid store name' in result.output.lower()
    assert active.read_text() == 'default\n'


@pytest.fixture
def offline_doctor(monkeypatch):
    """Keep the real report path while isolating network and service probes."""
    monkeypatch.setattr(doctor_mod.sch, 'status', lambda: {
        'installed': False, 'state': 'STOPPED', 'platform': 'test',
        })
    for name in ('llm_probe', 'embed_probe'):
        monkeypatch.setattr(doctor_mod, f'check_{name}',
                            lambda name=name: {
                                'name': name, 'status': 'pass', 'detail': {}})


@pytest.mark.parametrize(('state', 'expected'), [
    ('absent', 'pass'), ('foreign-directory', 'pass'),
    ('foreign-link', 'pass'), ('healthy', 'pass'),
    ('dangling', 'fail'), ('missing-file', 'fail'),
    ('unreadable', 'fail'), ('invalid-utf8', 'fail'),
])
def test_doctor_reports_codex_skill_health_through_cli(
        monkeypatch, tmp_path, mm_runner, offline_doctor, state, expected):
    """Verify doctor grades each Codex skill state and exits 1 on a failure.

    Mutation: check_codex_skill passes a dangling link or an unreadable,
    missing, or non-UTF-8 SKILL.md, or grades a foreign link or directory as
    broken.
    Oracle: Hand-built link states with a hand-written status per row; the JSON
    and --text output, exit code, and remediation text.
    """
    link = Path.home() / '.agents/skills/memman'
    link.parent.mkdir(parents=True)
    skill = link / 'SKILL.md'
    # Custom CODEX_HOME does not relocate the user-scoped skills directory.
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'custom-codex'))
    if state == 'foreign-directory':
        link.mkdir()
    elif state == 'foreign-link':
        link.symlink_to(tmp_path / 'foreign-missing-target')
    elif state != 'absent':
        target = tmp_path / 'old-env/memman/setup/assets/codex'
        link.symlink_to(target)
        if state != 'dangling':
            target.mkdir(parents=True)
            if state != 'missing-file':
                skill.write_bytes(b'\xff' if state == 'invalid-utf8' else
                                  b'---\nname: memman\n---\nMemory instructions\n')
        if state == 'unreadable':
            original = Path.read_text

            def refuse_skill(path, *args, **kwargs):
                if path == skill:
                    raise PermissionError('skill access denied')
                return original(path, *args, **kwargs)

            monkeypatch.setattr(Path, 'read_text', refuse_skill)

    report = invoke(mm_runner, ['doctor'])
    assert report.exit_code == (1 if expected == 'fail' else 0), report.output
    payload = json.loads(report.stdout)
    check = next(row for row in payload['checks'] if row['name'] == 'codex_skill')
    assert check['status'] == expected
    assert check['detail']['skill'] == str(skill)
    if expected == 'fail':
        assert payload['status'] == 'fail'
        assert 'memman install --codex' in check['detail']['remediation']
        text_report = invoke(mm_runner, ['doctor', '--text'])
        assert text_report.exit_code == 1
        assert '[FAIL] codex_skill' in text_report.stdout
        assert 'memman install --codex' in text_report.stdout
    if state == 'absent':
        assert not link.exists()
    elif state == 'foreign-directory':
        assert list(link.iterdir()) == []
    else:
        assert link.is_symlink()


@pytest.mark.parametrize(('claude_state', 'expected'), [
    ('unrelated-settings', 'pass'), ('foreign-assets', 'pass'),
    ('owned-skill-only', 'warn'), ('owned-hook-only', 'warn'),
    ('registered-hook-only', 'warn'),
])
def test_codex_only_doctor_ignores_uninstalled_claude_integration(
        tmp_path, mm_runner, offline_doctor, claude_state, expected):
    """Verify doctor warns only on a partial owned Claude install.

    Mutation: check_claude_hooks drops its early return, so unrelated settings
    warn, or ignores owned assets and passes a partial install.
    Oracle: Hand-built settings and asset layouts with a hand-written status
    per row; a non-empty missing list on each warn row.
    """
    install_codex({'skills_dir': str(Path.home() / '.agents/skills'),
                   'config_dir': str(Path.home() / '.codex')})
    base = Path.home() / '.claude'
    base.mkdir()
    settings = base / 'settings.json'
    if claude_state in {'unrelated-settings', 'foreign-assets'}:
        settings.write_text(json.dumps({
            'theme': 'dark',
            'hooks': {'SessionStart': [{'hooks': [
                {'type': 'command', 'command': '/bin/true'}]}]},
            }))
        if claude_state == 'foreign-assets':
            foreign_skill = base / 'skills/memman/SKILL.md'
            foreign_skill.parent.mkdir(parents=True)
            foreign_skill.write_text('User-owned unrelated memory skill')
            (base / 'hooks/memman').mkdir(parents=True)
    elif claude_state in {'owned-skill-only', 'owned-hook-only'}:
        path = (base / 'skills/memman/SKILL.md'
                if claude_state == 'owned-skill-only'
                else base / 'hooks/memman/prime.sh')
        path.parent.mkdir(parents=True)
        # Even dangling owned assets indicate an incomplete Claude install.
        path.symlink_to(tmp_path / 'old-env/memman/setup/assets/claude' / path.name)
    else:
        script = base / 'hooks/memman/prime.sh'
        script.parent.mkdir(parents=True)
        script.write_text('#!/bin/sh\n')
        settings.write_text(json.dumps({
            'hooks': {'SessionStart': [{'hooks': [{
                'type': 'command', 'command': str(script)}]}]},
            }))

    report = invoke(mm_runner, ['doctor'])
    assert report.exit_code == 0, report.output
    checks = {row['name']: row for row in json.loads(report.stdout)['checks']}
    assert checks['codex_skill']['status'] == 'pass'
    assert checks['claude_hooks']['status'] == expected
    if expected == 'warn':
        assert checks['claude_hooks']['detail']['missing']
