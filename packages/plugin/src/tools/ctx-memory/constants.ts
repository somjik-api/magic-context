export const CTX_MEMORY_TOOL_NAME = "ctx_memory";
export const CTX_MEMORY_DESCRIPTION = `Durable project knowledge shared across every session on this project.

Your active memories are already visible in <project-memory> (each with its id), and every future session starts with them — write one when you learn something future sessions must know: a project rule, an architectural fact, a hard-won constraint, a config value, or a naming convention. Keep each memory one standalone fact, phrased to make sense without this session's context.

Actions:
- write: save a new memory (content + category).
- update: rewrite one memory whose fact changed (ids: [one], content).
- archive: retire wrong or obsolete memories (ids: [one or more], optional reason).
- merge: collapse duplicates into one memory (ids: [two or more], content).
- get: fetch memories by id (ids: [1-20]); readable in every status.
- list: page through stored memories (dreamer-only). Paginated — it returns at most one bounded page and, when more memories exist, prints a footer with the exact ctx_memory(action="list", offset=…) call to fetch the next page. Page through with offset until the footer disappears — never assume the first page is the full set.

Example: ctx_memory(action="write", category="CONSTRAINTS", content="Pi stores sessions as JSONL under ~/.pi/agent/sessions/, not SQLite")`;
// Default page size for `list` when the caller does not specify a limit.
// Large enough for maintenance tasks to page through the full memory set, but
// still bounded so a single page stays small.
export const DEFAULT_LIST_LIMIT = 100;

// Hard char budget for a single `list` page result. A project can accumulate
// thousands of active memories (hundreds of KB); returning them all in one
// tool result overflows the dreamer model's context window and deadlocks the
// consolidate/verify/archive tasks. The list handler stops adding rows once
// this budget is reached and tells the caller to request the next page via
// `offset`, so a single call can never overflow regardless of total volume.
export const LIST_PAGE_CHAR_BUDGET = 24_000;
