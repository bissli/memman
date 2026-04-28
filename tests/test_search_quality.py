"""Content quality pattern detection tests."""

from memman.search.quality import check_content_quality


class TestInstanceIdDetected:
    """AWS instance IDs trigger warnings."""

    def test_instance_id_detected(self):
        """AWS instance ID pattern matched."""
        w = check_content_quality('Deployed i-0c220c2402a5245bc')
        assert 'AWS instance ID' in w


class TestResourceCountDetected:
    """Resource count language triggers warning."""

    def test_resource_count_detected(self):
        """'N resources total' pattern matched."""
        w = check_content_quality('32 resources total in the stack')
        assert 'resource count' in w

    def test_singular_resource(self):
        """'1 resource total' also matched."""
        w = check_content_quality('1 resource total')
        assert 'resource count' in w


class TestVerificationReceipt:
    """Verification language triggers warning."""

    def test_all_verified(self):
        """'All drives verified' pattern matched."""
        w = check_content_quality('All drives verified: D: 2500GB')
        assert 'verification receipt' in w

    def test_every_verified(self):
        """'every ... verified' variant matched."""
        w = check_content_quality('Every instance verified healthy')
        assert 'verification receipt' in w


class TestStateObservation:
    """State observation language triggers warning."""

    def test_state_clean(self):
        """'state clean' pattern matched."""
        w = check_content_quality('Terraform state clean after apply')
        assert 'state observation' in w

    def test_state_is_clean(self):
        """'state is clean' variant matched."""
        w = check_content_quality('State is clean')
        assert 'state observation' in w


class TestDeploymentReceipt:
    """Deployment receipt language triggers warning."""

    def test_deployed_via(self):
        """'deployed via' pattern matched."""
        w = check_content_quality('Stack deployed via Terraform')
        assert 'deployment receipt' in w

    def test_applied_via(self):
        """'applied via' variant matched."""
        w = check_content_quality('Changes applied via CI pipeline')
        assert 'deployment receipt' in w


class TestLineNumberReference:
    """Line number references trigger warnings."""

    def test_line_number(self):
        """'line 42' pattern matched."""
        w = check_content_quality('Error on line 42 of the module')
        assert 'line number reference' in w

    def test_line_number_case_insensitive(self):
        """'Line 100' case-insensitive match."""
        w = check_content_quality('Line 100 has the bug')
        assert 'line number reference' in w


class TestLineCount:
    """Line count references trigger warnings."""

    def test_line_count(self):
        """'4841 lines' pattern matched."""
        w = check_content_quality('The file grew to 4841 lines')
        assert 'line count' in w

    def test_two_digit_line_count(self):
        """'50 lines' also matched."""
        w = check_content_quality('Function is 50 lines long')
        assert 'line count' in w

    def test_single_digit_no_match(self):
        """'3 lines' should not match (single digit)."""
        w = check_content_quality('Only 3 lines of config')
        assert 'line count' not in w


class TestSymbolLineReference:
    """Function:line-number references trigger warnings."""

    def test_function_line_ref(self):
        """'main:28' pattern matched."""
        w = check_content_quality('See main:28 for the entry point')
        assert 'function/symbol line reference' in w

    def test_long_symbol_ref(self):
        """'import_issuer_data:121' pattern matched."""
        w = check_content_quality(
            'Fixed import_issuer_data:121 off-by-one')
        assert 'function/symbol line reference' in w

    def test_single_digit_no_match(self):
        """'port:5' should not match (single digit)."""
        w = check_content_quality('Set port:5 for debugging')
        assert 'function/symbol line reference' not in w


class TestLineNumberCorrection:
    """Arrow-style line corrections trigger warnings."""

    def test_arrow_correction(self):
        """'422→421' pattern matched."""
        w = check_content_quality('Line changed 422→421 after edit')
        assert 'line number correction' in w


class TestBackReference:
    """Cross-insight references trigger warnings."""

    def test_memory_bracketed_index(self):
        """'memory [3]' pattern matched."""
        w = check_content_quality('Aligns with memory [3] on retries')
        assert 'back-reference' in w

    def test_memories_plural(self):
        """'memories [0]' plural variant matched."""
        w = check_content_quality('See memories [0] and [4] for context')
        assert 'back-reference' in w

    def test_memory_no_brackets_no_match(self):
        """Plain word 'memory' should not match."""
        w = check_content_quality('Volatile memory is cleared on reboot')
        assert 'back-reference' not in w


class TestUppercaseSectionHeader:
    """All-caps section markers inside content trigger warnings."""

    def test_root_cause_header(self):
        """'ROOT CAUSE: ...' pattern matched."""
        w = check_content_quality(
            'Outage observed. ROOT CAUSE: misconfigured timeout.')
        assert 'uppercase section header' in w

    def test_key_finding_header(self):
        """'KEY FINDING: ...' pattern matched."""
        w = check_content_quality('KEY FINDING: replicas were stale')
        assert 'uppercase section header' in w

    def test_short_acronym_no_match(self):
        """'URL: https://...' should not match (too short)."""
        w = check_content_quality('See URL: https://example.com')
        assert 'uppercase section header' not in w

    def test_json_colon_no_match(self):
        """'JSON:' four-letter acronym should not match."""
        w = check_content_quality('Returned JSON: with the data')
        assert 'uppercase section header' not in w

    def test_no_space_after_colon_no_match(self):
        """'FOO_BAR:value' without trailing space should not match."""
        w = check_content_quality('Set ENV_VAR:production for the run')
        assert 'uppercase section header' not in w


class TestTransientTimeMarker:
    """'currently' word triggers warning."""

    def test_currently(self):
        """'currently' lowercase matched."""
        w = check_content_quality('The pipeline currently runs hourly')
        assert 'transient time marker' in w

    def test_currently_capitalized(self):
        """'Currently' at sentence start matched."""
        w = check_content_quality('Currently the queue is empty')
        assert 'transient time marker' in w

    def test_concurrent_no_match(self):
        """'concurrent' should not match (not a whole word)."""
        w = check_content_quality('Concurrent writes are serialized')
        assert 'transient time marker' not in w


class TestDatedObservation:
    """'as of YYYY-MM-DD' triggers warning."""

    def test_iso_date(self):
        """'as of 2026-04-28' matched."""
        w = check_content_quality(
            'Throughput is 4 req/s as of 2026-04-28')
        assert 'dated observation' in w

    def test_case_insensitive(self):
        """'AS OF 2026-04-28' matched case-insensitively."""
        w = check_content_quality('AS OF 2026-04-28 nothing has changed')
        assert 'dated observation' in w

    def test_no_date_no_match(self):
        """'as of last week' (no ISO date) should not match."""
        w = check_content_quality('Stable as of last week')
        assert 'dated observation' not in w


class TestCleanContentNoWarnings:
    """Durable reasoning produces no warnings."""

    def test_durable_fact(self):
        """Platform behavior insight triggers nothing."""
        w = check_content_quality(
            'EC2Launch v2 does not re-run userdata by default')
        assert w == []

    def test_architectural_decision(self):
        """Design decision triggers nothing."""
        w = check_content_quality(
            'Chose SQLite over Postgres for single-node simplicity')
        assert w == []

    def test_user_preference(self):
        """User preference triggers nothing."""
        w = check_content_quality(
            'User prefers snake_case for all variable names')
        assert w == []


class TestNoFalsePositives:
    """Good entries from tradar DB produce zero warnings."""

    def test_ebsnvme_entry(self):
        """ebsnvme-id /dev/ prefix entry — no warnings."""
        w = check_content_quality(
            'ebsnvme-id outputs device paths with /dev/ prefix')
        assert w == []

    def test_rds_sa_entry(self):
        """RDS sa not sysadmin entry — no warnings."""
        w = check_content_quality(
            'Cannot grant sysadmin to sa in RDS SQL Server')
        assert w == []

    def test_quicksetup_ssm(self):
        """QuickSetup SSM entry — no warnings."""
        w = check_content_quality(
            'QuickSetup SSM duplicates via CloudFormation stacks')
        assert w == []

    def test_port_number_no_false_positive(self):
        """'port:5' should not trigger symbol line reference."""
        w = check_content_quality('Connect to port:5 for the service')
        assert 'function/symbol line reference' not in w

    def test_localhost_port(self):
        """'localhost:8080' should not trigger symbol line reference."""
        w = check_content_quality('Server running on localhost:8080')
        assert 'function/symbol line reference' not in w

    def test_postgres_port(self):
        """'postgres:5432' should not trigger symbol line reference."""
        w = check_content_quality('PostgreSQL on port:5432')
        assert 'function/symbol line reference' not in w

    def test_python_version(self):
        """'python:3.11' should not trigger symbol line reference."""
        w = check_content_quality('Using python:3.11 base image')
        assert 'function/symbol line reference' not in w

    def test_docker_tag(self):
        """'alpine:3.18' should not trigger symbol line reference."""
        w = check_content_quality('FROM alpine:3.18 in Dockerfile')
        assert 'function/symbol line reference' not in w

    def test_node_version(self):
        """'node:20' should not trigger symbol line reference."""
        w = check_content_quality('docker pull node:20')
        assert 'function/symbol line reference' not in w

    def test_redis_port(self):
        """'redis:6379' should not trigger symbol line reference."""
        w = check_content_quality('Connect to redis:6379 in compose')
        assert 'function/symbol line reference' not in w

    def test_baseline_no_false_positive(self):
        """'baseline' should not trigger line number reference."""
        w = check_content_quality('baseline performance improved 20%')
        assert 'line number reference' not in w

    def test_deadline_no_false_positive(self):
        """'deadline' should not trigger line number reference."""
        w = check_content_quality('deadline for milestone is next week')
        assert 'line number reference' not in w

    def test_pipeline_no_false_positive(self):
        """'pipeline' should not trigger line number reference."""
        w = check_content_quality('CI pipeline runs on every commit')
        assert 'line number reference' not in w


class TestMultipleWarnings:
    """Content with multiple transient patterns returns all warnings."""

    def test_multiple_patterns(self):
        """Deployment receipt with instance IDs triggers multiple warnings."""
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
