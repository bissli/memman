export default function register(api) {
  // api.pluginConfig holds plugins.entries.memman.config from openclaw.json
  const cfg = api.pluginConfig ?? {};
  const remind  = cfg.remind  !== false; // default on

  // --- before_prompt_build ---
  // Always load the memman skill; optionally inject the recall hint.
  // Notes:
  // - No write hint rides this event. before_prompt_build runs before
  //   the exchange exists, so a nudge to store what the exchange
  //   settled asks for a judgment the agent cannot yet make.
  // - The write contract reaches the agent once, at agent:bootstrap,
  //   where the memman-prime hook injects `memman guide`.
  api.on("before_prompt_build", async () => {
    const parts = ["[memman] load memman skill"];

    if (remind) {
      parts.push("[memman] Evaluate: recall needed?");
    }

    return { prependContext: parts.join("\n\n") };
  });
}
