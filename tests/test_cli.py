"""Tests for mnemon.cli — Click CLI commands via CliRunner."""

import json
import pathlib

import pytest
from click.testing import CliRunner
from mnemon.cli import cli


@pytest.fixture
def runner(tmp_path):
    """CliRunner with --data-dir pointing to temp directory."""
    r = CliRunner()
    data_dir = str(tmp_path / 'mnemon_data')
    pathlib.Path(data_dir).mkdir(exist_ok=True, parents=True)
    return r, data_dir


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    """Prevent CLI tests from making real HTTP requests to Ollama."""
    monkeypatch.setattr(
        'mnemon.embed.ollama.Client.available', lambda self: False)


def invoke(runner_tuple, args):
    """Helper to invoke CLI with data-dir."""
    r, data_dir = runner_tuple
    return r.invoke(cli, ['--data-dir', data_dir] + args)


def test_remember_basic(runner):
    """Store a basic insight."""
    result = invoke(runner, ['remember', 'Go uses SQLite', '--no-diff'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['action'] in {'added', 'updated'}
    assert data['content'] == 'Go uses SQLite'


def test_remember_with_flags(runner):
    """Store with category, importance, tags."""
    result = invoke(runner, [
        'remember', 'Use Docker', '--no-diff',
        '--cat', 'decision', '--imp', '4',
        '--tags', 'docker,deployment'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['category'] == 'decision'
    assert data['importance'] == 4


def test_remember_invalid_category(runner):
    """Invalid category is rejected."""
    result = invoke(runner, ['remember', 'test', '--cat', 'bogus'])
    assert result.exit_code != 0


def test_remember_invalid_importance(runner):
    """Importance outside 1-5 is rejected."""
    result = invoke(runner, ['remember', 'test', '--imp', '0'])
    assert result.exit_code != 0


def test_recall_basic(runner):
    """Recall after remembering."""
    invoke(runner, ['remember', 'Go uses SQLite for storage', '--no-diff'])
    result = invoke(runner, ['recall', 'Go SQLite'])
    assert result.exit_code == 0


def test_recall_basic_mode(runner):
    """Basic recall returns array."""
    invoke(runner, ['remember', 'Go uses SQLite', '--no-diff'])
    result = invoke(runner, ['recall', 'Go', '--basic'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_search_basic(runner):
    """Search returns scored results."""
    invoke(runner, ['remember', 'Go uses SQLite for storage', '--no-diff'])
    result = invoke(runner, ['search', 'Go SQLite'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)


def test_forget_basic(runner):
    """Forget an insight by ID."""
    result = invoke(runner, ['remember', 'to be forgotten', '--no-diff'])
    data = json.loads(result.output)
    iid = data['id']
    result = invoke(runner, ['forget', iid])
    assert result.exit_code == 0
    fdata = json.loads(result.output)
    assert fdata['status'] == 'deleted'


def test_forget_writes_oplog(runner):
    """Forget command writes an oplog entry atomically."""
    result = invoke(runner, ['remember', 'oplog test content', '--no-diff'])
    data = json.loads(result.output)
    iid = data['id']
    invoke(runner, ['forget', iid])
    result = invoke(runner, ['log', '--stats'])
    assert result.exit_code == 0
    log_data = json.loads(result.output)
    assert 'forget' in log_data['operation_counts']


def test_forget_nonexistent_fails(runner):
    """Forget with nonexistent ID returns error."""
    result = invoke(runner, ['forget', 'nonexistent-id-12345'])
    assert result.exit_code != 0


def test_store_list(runner):
    """Store list shows stores."""
    result = invoke(runner, ['store', 'list'])
    assert result.exit_code == 0


def test_store_create(runner):
    """Create a new store."""
    result = invoke(runner, ['store', 'create', 'test-store'])
    assert result.exit_code == 0
    assert 'Created' in result.output


def test_store_create_duplicate(runner):
    """Duplicate store name is rejected."""
    invoke(runner, ['store', 'create', 'dup'])
    result = invoke(runner, ['store', 'create', 'dup'])
    assert result.exit_code != 0


def test_store_set(runner):
    """Set active store."""
    invoke(runner, ['store', 'create', 'work'])
    result = invoke(runner, ['store', 'set', 'work'])
    assert result.exit_code == 0
    assert 'Active store' in result.output


def test_store_remove(runner):
    """Remove a non-active store."""
    invoke(runner, ['store', 'create', 'temp'])
    result = invoke(runner, ['store', 'remove', 'temp'])
    assert result.exit_code == 0
    assert 'Removed' in result.output


def test_status_basic(runner):
    """Status returns JSON."""
    invoke(runner, ['remember', 'test insight', '--no-diff'])
    result = invoke(runner, ['status'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'total_insights' in data


def test_log_basic(runner):
    """Log shows recent operations."""
    invoke(runner, ['remember', 'test insight', '--no-diff'])
    result = invoke(runner, ['log'])
    assert result.exit_code == 0


def test_gc_suggest(runner):
    """GC suggest mode returns JSON."""
    invoke(runner, ['remember', 'test insight', '--no-diff'])
    result = invoke(runner, ['gc'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert 'total_insights' in data


def test_viz_dot(runner):
    """Viz dot format renders."""
    invoke(runner, ['remember', 'test insight', '--no-diff'])
    result = invoke(runner, ['viz'])
    assert result.exit_code == 0
    assert 'digraph' in result.output


def test_js_str_escapes_script_close():
    """_js_str prevents XSS via </script> in content."""
    from mnemon.cli import _js_str
    result = _js_str('</script><script>alert(1)</script>')
    assert '</script>' not in result
    assert '<\\/' in result


def test_js_str_normal_string():
    """_js_str handles regular strings correctly."""
    from mnemon.cli import _js_str
    result = _js_str('hello world')
    assert result == '"hello world"'


def test_remember_quality_warnings(runner):
    """Content with 2+ quality warnings is rejected."""
    result = invoke(runner, [
        'remember', 'i-0c220c2402a5245bc deployed via Terraform',
        '--no-diff'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['action'] == 'rejected'
    assert 'AWS instance ID' in data['quality_warnings']
    assert 'deployment receipt' in data['quality_warnings']


def test_remember_no_quality_warnings(runner):
    """Durable content produces empty quality_warnings."""
    result = invoke(runner, [
        'remember', 'SQLite chosen for single-node simplicity',
        '--no-diff'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['quality_warnings'] == []


def test_gc_review(runner):
    """GC --review flags transient content with 1 warning (stored)."""
    invoke(runner, [
        'remember', 'i-0c220c2402a5245bc shows the issue',
        '--no-diff'])
    invoke(runner, [
        'remember', 'SQLite chosen for simplicity', '--no-diff'])
    result = invoke(runner, ['gc', '--review'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['total_flagged'] == 1
    assert len(data['review_results']) == 1
    assert 'AWS instance ID' in data['review_results'][0]['quality_warnings']


def test_remember_quality_rejection(runner):
    """2+ warnings rejected, 1 warning stored, quality-reject in oplog."""
    result = invoke(runner, [
        'remember', 'Stack deployed via Terraform. 32 resources total.',
        '--no-diff'])
    data = json.loads(result.output)
    assert data['action'] == 'rejected'
    assert len(data['quality_warnings']) >= 2

    result = invoke(runner, [
        'remember', 'i-0c220c2402a5245bc shows the issue',
        '--no-diff'])
    data = json.loads(result.output)
    assert data['action'] == 'added'
    assert len(data['quality_warnings']) == 1

    result = invoke(runner, ['log', '--limit', '10'])
    assert 'quality-reject' in result.output


def test_replace_quality_rejection(runner):
    """Replace path also rejects content with 2+ quality warnings."""
    result = invoke(runner, ['remember', 'original fact', '--no-diff'])
    old_id = json.loads(result.output)['id']

    result = invoke(runner, [
        'replace', old_id,
        'Stack deployed via Terraform. 32 resources total.'])
    data = json.loads(result.output)
    assert data['action'] == 'rejected'
    assert len(data['quality_warnings']) >= 2


def test_gc_review_empty(runner):
    """GC --review with clean store returns zero flagged."""
    invoke(runner, [
        'remember', 'SQLite chosen for simplicity', '--no-diff'])
    result = invoke(runner, ['gc', '--review'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['total_flagged'] == 0


def test_replace_basic(runner):
    """Replace an insight, verify old soft-deleted, new exists."""
    result = invoke(runner, ['remember', 'old fact', '--no-diff',
                             '--cat', 'fact', '--imp', '3'])
    old_id = json.loads(result.output)['id']

    result = invoke(runner, ['replace', old_id, 'new fact'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['action'] == 'replaced'
    assert data['replaced_id'] == old_id
    assert data['content'] == 'new fact'

    result = invoke(runner, ['recall', 'new fact', '--basic'])
    hits = json.loads(result.output)
    assert any(h['content'] == 'new fact' for h in hits)

    result = invoke(runner, ['recall', 'old fact', '--basic'])
    hits = json.loads(result.output)
    assert not any(h['content'] == 'old fact' for h in hits)


def test_replace_inherits_metadata(runner):
    """Replace without flags inherits cat/imp/tags from original."""
    result = invoke(runner, ['remember', 'original insight', '--no-diff',
                             '--cat', 'decision', '--imp', '5',
                             '--tags', 'arch,design'])
    old_id = json.loads(result.output)['id']

    result = invoke(runner, ['replace', old_id, 'updated insight'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['category'] == 'decision'
    assert data['importance'] == 5
    assert 'arch' in data['tags']
    assert 'design' in data['tags']


def test_replace_overrides_metadata(runner):
    """Replace with explicit flags uses new values."""
    result = invoke(runner, ['remember', 'original', '--no-diff',
                             '--cat', 'fact', '--imp', '2'])
    old_id = json.loads(result.output)['id']

    result = invoke(runner, ['replace', old_id, 'updated',
                             '--cat', 'decision', '--imp', '5'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['category'] == 'decision'
    assert data['importance'] == 5


def test_replace_preserves_access_count(runner):
    """Replace carries over access_count from original."""
    result = invoke(runner, ['remember', 'accessed insight', '--no-diff'])
    old_id = json.loads(result.output)['id']
    invoke(runner, ['recall', 'accessed', '--basic'])
    invoke(runner, ['recall', 'accessed', '--basic'])

    result = invoke(runner, ['replace', old_id, 'replacement'])
    assert result.exit_code == 0
    new_id = json.loads(result.output)['id']

    result = invoke(runner, ['recall', 'replacement', '--basic'])
    hits = json.loads(result.output)
    match = [h for h in hits if h['id'] == new_id]
    assert match
    assert match[0]['access_count'] >= 2


def test_replace_nonexistent_id(runner):
    """Replace a nonexistent ID produces error."""
    result = invoke(runner, ['replace', 'nonexistent-id', 'content'])
    assert result.exit_code != 0
    assert 'not found' in result.output


def test_replace_already_deleted(runner):
    """Replace an already-deleted insight produces error."""
    result = invoke(runner, ['remember', 'to delete', '--no-diff'])
    old_id = json.loads(result.output)['id']
    invoke(runner, ['forget', old_id])

    result = invoke(runner, ['replace', old_id, 'new content'])
    assert result.exit_code != 0
    assert 'not found' in result.output


def test_replace_oplog_entries(runner):
    """Replace logs both replace and remember ops."""
    result = invoke(runner, ['remember', 'oplog test', '--no-diff'])
    old_id = json.loads(result.output)['id']

    result = invoke(runner, ['replace', old_id, 'oplog replacement'])
    new_id = json.loads(result.output)['id']

    result = invoke(runner, ['log', '--limit', '10'])
    assert result.exit_code == 0
    assert 'replace' in result.output
    assert 'remember' in result.output


def test_link_creates_both_directions(runner):
    """Link creates edges in both directions atomically."""
    r1 = invoke(runner, ['remember', 'first insight', '--no-diff'])
    id1 = json.loads(r1.output)['id']
    r2 = invoke(runner, ['remember', 'second insight', '--no-diff'])
    id2 = json.loads(r2.output)['id']

    result = invoke(runner, [
        'link', id1, id2, '--type', 'causal'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data['status'] == 'linked'

    fwd = invoke(runner, ['related', id1, '--edge', 'causal'])
    assert fwd.exit_code == 0
    fwd_data = json.loads(fwd.output)
    assert any(e['id'] == id2 for e in fwd_data)

    rev = invoke(runner, ['related', id2, '--edge', 'causal'])
    assert rev.exit_code == 0
    rev_data = json.loads(rev.output)
    assert any(e['id'] == id1 for e in rev_data)


def test_link_meta_non_dict_fails(runner):
    """Non-dict JSON metadata is rejected."""
    r1 = invoke(runner, ['remember', 'first', '--no-diff'])
    id1 = json.loads(r1.output)['id']
    r2 = invoke(runner, ['remember', 'second', '--no-diff'])
    id2 = json.loads(r2.output)['id']

    result = invoke(runner, [
        'link', id1, id2, '--type', 'semantic',
        '--meta', '[1, 2]'])
    assert result.exit_code != 0
    assert 'object' in result.output.lower() or 'dict' in result.output.lower()


def test_link_self_edge_rejected(runner):
    """Linking an insight to itself is rejected."""
    r1 = invoke(runner, ['remember', 'self-link test', '--no-diff'])
    id1 = json.loads(r1.output)['id']

    result = invoke(runner, ['link', id1, id1, '--type', 'semantic'])
    assert result.exit_code != 0
    assert 'itself' in result.output.lower()


def test_recall_source_filter_smart(runner):
    """Smart recall respects --source filter."""
    invoke(runner, ['remember', 'Go uses SQLite for storage',
                    '--no-diff', '--source', 'agent'])
    invoke(runner, ['remember', 'Python uses PostgreSQL for storage',
                    '--no-diff', '--source', 'human'])

    result = invoke(runner, ['recall', 'uses storage', '--source', 'agent'])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for r in data['results']:
        assert r['insight']['source'] == 'agent'
