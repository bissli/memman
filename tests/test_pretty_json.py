"""`--pretty` indents a command's JSON reply for terminal reading.
"""

import json

from tests.conftest import invoke


def test_pretty_after_subcommand_indents_json(mm_runner):
    """Verify a trailing `--pretty` indents the reply and keeps its content.

    Mutation: `--pretty` ignored, or rejected as an unknown option after
        the subcommand.
    Oracle: the same command's one-line reply, parsed.
    """
    plain = invoke(mm_runner, ['store', 'list'])
    pretty = invoke(mm_runner, ['store', 'list', '--pretty'])
    assert pretty.exit_code == 0, pretty.output
    assert '\n  "active"' in pretty.output
    assert json.loads(pretty.output) == json.loads(plain.output)


def test_pretty_after_double_dash_stays_an_argument(mm_runner):
    """Verify a `--pretty` token after `--` reaches the command as a value.

    Mutation: stripping every `--pretty` token, so `store create` loses
        its name and fails on a missing argument.
    Oracle: the invalid-store-name error naming `--pretty`.
    """
    result = invoke(mm_runner, ['store', 'create', '--', '--pretty'])
    assert "invalid store name '--pretty'" in result.output
