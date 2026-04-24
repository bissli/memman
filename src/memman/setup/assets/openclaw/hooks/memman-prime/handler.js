import { execSync } from "child_process";

const handler = async (event) => {
  if (event.type !== "agent" || event.action !== "bootstrap") return;

  const parts = [];

  try {
    const status = execSync("memman status 2>/dev/null", {
      timeout: 3000,
      encoding: "utf-8",
    });
    if (status) {
      const insights =
        status.match(/"total_insights":\s*(\d+)/)?.[1] || "0";
      const edges = status.match(/"edge_count":\s*(\d+)/)?.[1] || "0";
      parts.push(
        `[memman] Memory active (${insights} insights, ${edges} edges).`
      );
    }
  } catch {
    parts.push("[memman] Memory active.");
  }

  try {
    const guide = execSync("memman guide 2>/dev/null", {
      timeout: 3000,
      encoding: "utf-8",
    });
    if (guide) parts.push(guide);
  } catch {
    // memman not on PATH — skill-only mode, no guide injection
  }

  if (parts.length === 0) return;

  event.context.bootstrapFiles.push({
    name: "MEMMAN-GUIDE.md",
    path: "memman/guide.md",
    content: parts.join("\n\n"),
    missing: false,
  });
};

export default handler;
