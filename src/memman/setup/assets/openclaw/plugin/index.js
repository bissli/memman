export default function register(api) {
  // api.pluginConfig holds plugins.entries.memman.config from openclaw.json
  const cfg = api.pluginConfig ?? {};
  const remind  = cfg.remind  !== false; // default on
  const nudge   = cfg.nudge   !== false; // default on

  // ── before_prompt_build ───────────────────────────────────────
  // Always load the memman skill; optionally inject remind/nudge hints.
  api.on("before_prompt_build", async () => {
    const parts = ["[memman] load memman skill"];

    if (remind) {
      parts.push(
        "[memman] Evaluate: recall needed? After responding, evaluate: remember needed?"
      );
    }

    if (nudge) {
      parts.push(
        "[memman] Consider: does this exchange warrant a `memman remember` call?"
      );
    }

    return { prependContext: parts.join("\n\n") };
  });
}
