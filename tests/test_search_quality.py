"""Content quality pattern detection tests.
"""

from memman.search.quality import check_content_quality


class TestInstanceIdDetected:
    """AWS instance IDs trigger warnings.
    """

    def test_instance_id_detected(self):
        """Verify a 17-hex-digit EC2 instance id is flagged.

        Mutation: the id regex requiring 18 hex digits, or dropping the i-
            prefix match.
        Oracle: hand-written sentence that holds a real 17-digit instance id.
        """
        w = check_content_quality('Deployed i-0c220c2402a5245bc')
        assert 'AWS instance ID' in w


class TestResourceCountDetected:
    """Resource count language triggers warning.
    """

    def test_resource_count_detected(self):
        """Verify "N resources total" is flagged as a resource count.

        Mutation: the regex dropping the plural s match, or requiring a
            different trailing word.
        Oracle: hand-written sentence that states a plural resource count.
        """
        w = check_content_quality('32 resources total in the stack')
        assert 'resource count' in w

    def test_singular_resource(self):
        """Verify "1 resource total" is flagged as a resource count.

        Mutation: making the trailing s in "resources?" mandatory.
        Oracle: hand-written sentence that states a singular resource count.
        """
        w = check_content_quality('1 resource total')
        assert 'resource count' in w


class TestVerificationReceipt:
    """Verification language triggers warning.
    """

    def test_all_verified(self):
        """Verify "All ... verified" is flagged as a verification receipt.

        Mutation: dropping the "all" alternative from the pattern.
        Oracle: hand-written sentence that starts with "All" and ends in
            "verified".
        """
        w = check_content_quality('All drives verified: D: 2500GB')
        assert 'verification receipt' in w

    def test_every_verified(self):
        """Verify "Every ... verified" is flagged as a verification receipt.

        Mutation: dropping the "every" alternative from the pattern.
        Oracle: hand-written sentence that starts with "Every" and ends in
            "verified".
        """
        w = check_content_quality('Every instance verified healthy')
        assert 'verification receipt' in w


class TestStateObservation:
    """State observation language triggers warning.
    """

    def test_state_clean(self):
        """Verify "state clean" is flagged as a state observation.

        Mutation: requiring the optional "is" in the pattern.
        Oracle: hand-written sentence that contains "state clean" with no "is".
        """
        w = check_content_quality('Terraform state clean after apply')
        assert 'state observation' in w

    def test_state_is_clean(self):
        """Verify "state is clean" is flagged, in any letter case.

        Mutation: dropping the "(?:is )?" branch or the IGNORECASE flag.
        Oracle: hand-written sentence that opens with a capitalized "State is
            clean".
        """
        w = check_content_quality('State is clean')
        assert 'state observation' in w


class TestDeploymentReceipt:
    """Deployment receipt language triggers warning.
    """

    def test_deployed_via(self):
        """Verify "deployed via" is flagged as a deployment receipt.

        Mutation: dropping "deployed" from the alternation.
        Oracle: hand-written sentence that contains "deployed via".
        """
        w = check_content_quality('Stack deployed via Terraform')
        assert 'deployment receipt' in w

    def test_applied_via(self):
        """Verify "applied via" is flagged as a deployment receipt.

        Mutation: dropping "applied" from the alternation.
        Oracle: hand-written sentence that contains "applied via".
        """
        w = check_content_quality('Changes applied via CI pipeline')
        assert 'deployment receipt' in w


class TestLineNumberReference:
    """Line number references trigger warnings.
    """

    def test_line_number(self):
        """Verify "line N" is flagged as a line number reference.

        Mutation: requiring a digit run of three or more, or dropping the
            pattern.
        Oracle: hand-written sentence that contains "line 42".
        """
        w = check_content_quality('Error on line 42 of the module')
        assert 'line number reference' in w

    def test_line_number_case_insensitive(self):
        """Verify "Line N" at the start of text is flagged.

        Mutation: dropping IGNORECASE, or the lookbehind rejecting a match at
            position 0.
        Oracle: hand-written sentence that starts with capitalized "Line 100".
        """
        w = check_content_quality('Line 100 has the bug')
        assert 'line number reference' in w


class TestLineCount:
    """Line count references trigger warnings.
    """

    def test_line_count(self):
        """Verify a four-digit line count is flagged.

        Mutation: requiring exactly two digits before "lines".
        Oracle: hand-written sentence that contains "4841 lines".
        """
        w = check_content_quality('The file grew to 4841 lines')
        assert 'line count' in w

    def test_two_digit_line_count(self):
        r"""Verify a two-digit line count is flagged.

        Mutation: raising the minimum to three digits (\d{3,}).
        Oracle: hand-written sentence that contains "50 lines".
        """
        w = check_content_quality('Function is 50 lines long')
        assert 'line count' in w

    def test_single_digit_no_match(self):
        r"""Verify a one-digit line count is not flagged.

        Mutation: lowering the minimum to one digit (\d+ lines).
        Oracle: hand-written sentence that contains "3 lines", which sits just
            under the two-digit floor.
        """
        w = check_content_quality('Only 3 lines of config')
        assert 'line count' not in w


class TestSymbolLineReference:
    """Function:line-number references trigger warnings.
    """

    def test_function_line_ref(self):
        """Verify "name:NN" is flagged as a symbol line reference.

        Mutation: raising the digit floor above two, or excluding "main" as a
            name.
        Oracle: hand-written sentence that contains "main:28".
        """
        w = check_content_quality('See main:28 for the entry point')
        assert 'function/symbol line reference' in w

    def test_long_symbol_ref(self):
        r"""Verify a long underscored symbol with a line number is flagged.

        Mutation: narrowing \w+ so an underscore breaks the match.
        Oracle: hand-written sentence that contains "import_issuer_data:121".
        """
        w = check_content_quality(
            'Fixed import_issuer_data:121 off-by-one')
        assert 'function/symbol line reference' in w

    def test_single_digit_no_match(self):
        """Verify "name:N" with one digit is not flagged.

        Mutation: lowering the digit floor from two to one.
        Oracle: hand-written sentence that contains "port:5", which sits just
            under the two-digit floor.
        """
        w = check_content_quality('Set port:5 for debugging')
        assert 'function/symbol line reference' not in w


class TestLineNumberCorrection:
    """Arrow-style line corrections trigger warnings.
    """

    def test_arrow_correction(self):
        """Verify "N->M" written with an arrow is flagged as a correction.

        Mutation: dropping the arrow pattern, or matching an ASCII "->" instead
            of the arrow character.
        Oracle: hand-written sentence with the arrow character between two
            numbers
        """
        w = check_content_quality('Line changed 422\u2192421 after edit')
        assert 'line number correction' in w


class TestBackReference:
    """Cross-insight references trigger warnings.
    """

    def test_memory_bracketed_index(self):
        """Verify "memory [N]" is flagged as a back-reference.

        Mutation: dropping the bracketed-index part of the pattern.
        Oracle: hand-written sentence that contains "memory [3]".
        """
        w = check_content_quality('Aligns with memory [3] on retries')
        assert 'back-reference' in w

    def test_memories_plural(self):
        """Verify "memories [N]" is flagged as a back-reference.

        Mutation: matching only the singular "memory".
        Oracle: hand-written sentence that contains "memories [0]".
        """
        w = check_content_quality('See memories [0] and [4] for context')
        assert 'back-reference' in w

    def test_memory_no_brackets_no_match(self):
        """Verify the bare word "memory" is not flagged.

        Mutation: making the bracket group optional.
        Oracle: hand-written sentence that uses "memory" in the hardware sense
            with no index.
        """
        w = check_content_quality('Volatile memory is cleared on reboot')
        assert 'back-reference' not in w


class TestUppercaseSectionHeader:
    """All-caps section markers inside content trigger warnings.
    """

    def test_root_cause_header(self):
        """Verify "ROOT CAUSE: " mid-sentence is flagged as a header.

        Mutation: anchoring the pattern to the start of the text.
        Oracle: hand-written sentence that has "ROOT CAUSE: " after a sentence.
        """
        w = check_content_quality(
            'Outage observed. ROOT CAUSE: misconfigured timeout.')
        assert 'uppercase section header' in w

    def test_key_finding_header(self):
        """Verify "KEY FINDING: " at the start is flagged as a header.

        Mutation: requiring a leading sentence before the header.
        Oracle: hand-written sentence that starts with "KEY FINDING: ".
        """
        w = check_content_quality('KEY FINDING: replicas were stale')
        assert 'uppercase section header' in w

    def test_short_acronym_no_match(self):
        """Verify a short acronym label such as "URL:" is not flagged.

        Mutation: lowering the minimum header length from five characters.
        Oracle: hand-written sentence that contains "URL: ", three letters.
        """
        w = check_content_quality('See URL: https://example.com')
        assert 'uppercase section header' not in w

    def test_json_colon_no_match(self):
        """Verify a four-letter acronym label is not flagged.

        Mutation: lowering the minimum header length by one.
        Oracle: hand-written sentence that contains "JSON: ", four letters and
            one under the floor.
        """
        w = check_content_quality('Returned JSON: with the data')
        assert 'uppercase section header' not in w

    def test_no_space_after_colon_no_match(self):
        r"""Verify "NAME:value" with no space after the colon is not flagged.

        Mutation: relaxing the trailing \s+ to \s*.
        Oracle: hand-written sentence that contains "ENV_VAR:production" with
            no space.
        """
        w = check_content_quality('Set ENV_VAR:production for the run')
        assert 'uppercase section header' not in w


class TestTransientTimeMarker:
    """'currently' word triggers warning.
    """

    def test_currently(self):
        """Verify lower-case "currently" is flagged.

        Mutation: dropping the pattern.
        Oracle: hand-written sentence that contains "currently".
        """
        w = check_content_quality('The pipeline currently runs hourly')
        assert 'transient time marker' in w

    def test_currently_capitalized(self):
        """Verify capitalized "Currently" is flagged.

        Mutation: dropping IGNORECASE.
        Oracle: hand-written sentence that starts with "Currently".
        """
        w = check_content_quality('Currently the queue is empty')
        assert 'transient time marker' in w

    def test_concurrent_no_match(self):
        """Verify "Concurrent" is not flagged as "currently".

        Mutation: loosening the pattern to the stem 'curren'.
        Oracle: hand-written sentence that starts with "Concurrent".
        """
        w = check_content_quality('Concurrent writes are serialized')
        assert 'transient time marker' not in w


class TestDatedObservation:
    """'as of YYYY-MM-DD' triggers warning.
    """

    def test_iso_date(self):
        """Verify "as of YYYY-MM-DD" is flagged.

        Mutation: requiring a different date format.
        Oracle: hand-written sentence that contains "as of 2026-04-28".
        """
        w = check_content_quality(
            'Throughput is 4 req/s as of 2026-04-28')
        assert 'dated observation' in w

    def test_case_insensitive(self):
        """Verify "AS OF YYYY-MM-DD" is flagged in upper case.

        Mutation: dropping IGNORECASE.
        Oracle: hand-written sentence that has "AS OF 2026-04-28" in capitals.
        """
        w = check_content_quality('AS OF 2026-04-28 nothing has changed')
        assert 'dated observation' in w

    def test_no_date_no_match(self):
        """Verify "as of" without an ISO date is not flagged.

        Mutation: dropping the date digits from the pattern.
        Oracle: hand-written sentence that contains "as of last week".
        """
        w = check_content_quality('Stable as of last week')
        assert 'dated observation' not in w


class TestCleanContentNoWarnings:
    """Durable reasoning produces no warnings.
    """

    def test_durable_fact(self):
        """Verify a durable platform fact returns no warnings.

        Mutation: any pattern loosened until it fires on plain lower-case
            words.
        Oracle: hand-written sentence that holds a durable fact free of every
            transient marker.
        """
        w = check_content_quality(
            'EC2Launch v2 does not re-run userdata by default')
        assert w == []

    def test_architectural_decision(self):
        """Verify a design decision returns no warnings.

        Mutation: any pattern loosened until it fires on plain lower-case
            words.
        Oracle: hand-written sentence that holds a design rationale free of
            every transient marker.
        """
        w = check_content_quality(
            'Chose SQLite over Postgres for single-node simplicity')
        assert w == []

    def test_user_preference(self):
        """Verify a user preference returns no warnings.

        Mutation: the symbol or header pattern firing on snake_case words.
        Oracle: hand-written sentence that holds a preference with an
            underscored identifier.
        """
        w = check_content_quality(
            'User prefers snake_case for all variable names')
        assert w == []


class TestNoFalsePositives:
    """Good entries from tradar DB produce zero warnings.
    """

    def test_ebsnvme_entry(self):
        """Verify a device-path entry returns no warnings.

        Mutation: any pattern loosened until it fires on plain lower-case
            words.
        Oracle: hand-written sentence that holds a stored entry known to be
            durable.
        """
        w = check_content_quality(
            'ebsnvme-id outputs device paths with /dev/ prefix')
        assert w == []

    def test_rds_sa_entry(self):
        """Verify a short two-letter-name entry returns no warnings.

        Mutation: making the colon optional in the uppercase-header pattern.
        Oracle: hand-written sentence that holds a stored entry known to be
            durable.
        """
        w = check_content_quality(
            'Cannot grant sysadmin to sa in RDS SQL Server')
        assert w == []

    def test_quicksetup_ssm(self):
        """Verify a CamelCase and acronym entry returns no warnings.

        Mutation: loosening the deployment-receipt pattern to any word before
            'via'.
        Oracle: hand-written sentence that holds a stored entry known to be
            durable.
        """
        w = check_content_quality(
            'QuickSetup SSM duplicates via CloudFormation stacks')
        assert w == []

    def test_port_number_no_false_positive(self):
        """Verify "port:5" is not flagged as a symbol line reference.

        Mutation: dropping "port" from the negative lookahead.
        Oracle: hand-written sentence that contains "port:5".
        """
        w = check_content_quality('Connect to port:5 for the service')
        assert 'function/symbol line reference' not in w

    def test_localhost_port(self):
        """Verify "localhost:8080" is not flagged as a symbol line reference.

        Mutation: dropping "localhost" from the negative lookahead.
        Oracle: hand-written sentence that contains "localhost:8080".
        """
        w = check_content_quality('Server running on localhost:8080')
        assert 'function/symbol line reference' not in w

    def test_postgres_port(self):
        """Verify "port:5432" is not flagged as a symbol line reference.

        Mutation: dropping "port" from the negative lookahead.
        Oracle: hand-written sentence that contains "port:5432".
        """
        w = check_content_quality('PostgreSQL on port:5432')
        assert 'function/symbol line reference' not in w

    def test_python_version(self):
        """Verify "python:3.11" is not flagged as a symbol line reference.

        Mutation: dropping "python" from the negative lookahead.
        Oracle: hand-written sentence that contains "python:3.11".
        """
        w = check_content_quality('Using python:3.11 base image')
        assert 'function/symbol line reference' not in w

    def test_docker_tag(self):
        """Verify "alpine:3.18" is not flagged as a symbol line reference.

        Mutation: dropping "alpine" from the negative lookahead.
        Oracle: hand-written sentence that contains "alpine:3.18".
        """
        w = check_content_quality('FROM alpine:3.18 in Dockerfile')
        assert 'function/symbol line reference' not in w

    def test_node_version(self):
        """Verify "node:20" is not flagged as a symbol line reference.

        Mutation: dropping "node" from the negative lookahead.
        Oracle: hand-written sentence that contains "node:20".
        """
        w = check_content_quality('docker pull node:20')
        assert 'function/symbol line reference' not in w

    def test_redis_port(self):
        """Verify "redis:6379" is not flagged as a symbol line reference.

        Mutation: dropping "redis" from the negative lookahead.
        Oracle: hand-written sentence that contains "redis:6379".
        """
        w = check_content_quality('Connect to redis:6379 in compose')
        assert 'function/symbol line reference' not in w

    def test_baseline_no_false_positive(self):
        """Verify "baseline" is not flagged as "line N".

        Mutation: matching the bare substring 'line' with no number required.
        Oracle: hand-written sentence that contains "baseline" and a number
            elsewhere.
        """
        w = check_content_quality('baseline performance improved 20%')
        assert 'line number reference' not in w

    def test_deadline_no_false_positive(self):
        """Verify "deadline" is not flagged as "line N".

        Mutation: matching the bare substring 'line' with no number required.
        Oracle: hand-written sentence that contains "deadline".
        """
        w = check_content_quality('deadline for milestone is next week')
        assert 'line number reference' not in w

    def test_pipeline_no_false_positive(self):
        """Verify "pipeline" is not flagged as "line N".

        Mutation: matching the bare substring 'line' with no number required.
        Oracle: hand-written sentence that contains "pipeline".
        """
        w = check_content_quality('CI pipeline runs on every commit')
        assert 'line number reference' not in w


class TestMultipleWarnings:
    """Content with multiple transient patterns returns all warnings.
    """

    def test_multiple_patterns(self):
        """Verify each matching pattern adds exactly one warning.

        Mutation: returning after the first match, or duplicating a label.
        Oracle: hand-counted five labels for the five phrases in the input
        """
        content = (
            'TC-DB-01 (i-0c220c2402a5245bc) deployed via Terraform.'
            ' 32 resources total. All drives verified. State is clean.')
        w = check_content_quality(content)
        assert 'AWS instance ID' in w
        assert 'resource count' in w
        assert 'verification receipt' in w
        assert 'state observation' in w
        assert 'deployment receipt' in w
        assert len(w) == 5
