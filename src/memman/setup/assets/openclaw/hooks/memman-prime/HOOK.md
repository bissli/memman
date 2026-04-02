---
name: memman-prime
description: "Inject memman behavioral guide into agent bootstrap context"
metadata:
  openclaw:
    emoji: "🧠"
    events: ["agent:bootstrap"]
    export: "default"
    requires:
      bins: ["memman"]
---
