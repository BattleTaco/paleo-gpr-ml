---
name: feedback-write-in-michaels-voice
description: "Write all code, comments, notes, docs, and chat replies in Michael's own voice with zero AI tells; log everything important"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e6c235d4-3a76-4721-a696-dfcdc468b126
---

All code, comments, notes, docs, and chat replies for paleo-gpr-ml must read like Michael wrote them himself as an ML engineer, with zero AI tells. This is a hard requirement he called out explicitly.

**Why:** It is his sole-author research and his learning. The paper has to be his, and AI-sounding writing reads as inauthentic and undercuts credibility.

**No AI tells (ban list):**
- No em dashes or en dashes used as em dashes. Do not use "--" as a dash either. Use commas, periods, parentheses, "so", "but", "because".
- No "not just X, it's Y", no balanced tricolons, no "It's worth noting", "At its core", "Let's dive in", "Here's the thing", "That said".
- No filler hype words: leverage, robust, seamless, comprehensive, delve, crucial, vibrant.
- No formulaic paragraph openers or symmetric setups. Vary sentence length. Plain and direct.

**Sole author:** this is Michael's research alone. Write in the first person singular. Use "I", "my", "me", never the royal "we"/"our"/"us" in repo docs, notes, or code comments. I am working alongside him, but the writing is his.

**Voice to match:** his real writing in the Obsidian vault (Diary, Skill Sessions) is the ground truth: first person, direct, plain, a bit casual, explains his reasoning as he goes. For research notes keep it professional and structured but still plain and human. Note: the older `docs/notes/*.md` in the repo contain AI tells (em-dash style) and are NOT the style target anymore; strip those patterns going forward.

**Also:** write everything down (process, decisions, blockers) in `docs/notes/` or the Obsidian vault. Don't oversell. Report weak or negative results plainly. See [[research-priorities-paleo-gpr]], [[feedback-log-everything-we-do]], [[user-michael-ml-paleo]].
