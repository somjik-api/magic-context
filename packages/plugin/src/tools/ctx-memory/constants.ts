export const CTX_MEMORY_TOOL_NAME = "ctx_memory";
export const CTX_MEMORY_DESCRIPTION = `Manage cross-session project memories. Primary sessions can write new memories or delete stale ones. Dreamer sessions can also list, update, merge, and archive memories. Memories persist across sessions and are automatically injected into new sessions.

Supported actions: write, delete, list, update, merge, archive.

The list action is paginated: it returns at most one bounded page and, when more memories exist, prints a footer with the exact ctx_memory(action="list", offset=…) call to fetch the next page. Page through with offset until the footer disappears — never assume the first page is the full set.`;
export const DEFAULT_SEARCH_LIMIT = 10;

// Default page size for `list` when the caller does not specify a limit.
// Larger than DEFAULT_SEARCH_LIMIT because maintenance tasks page through the
// full memory set, but still bounded so a single page stays small.
export const DEFAULT_LIST_LIMIT = 100;

// Hard char budget for a single `list` page result. A project can accumulate
// thousands of active memories (hundreds of KB); returning them all in one
// tool result overflows the dreamer model's context window and deadlocks the
// consolidate/verify/archive tasks. The list handler stops adding rows once
// this budget is reached and tells the caller to request the next page via
// `offset`, so a single call can never overflow regardless of total volume.
export const LIST_PAGE_CHAR_BUDGET = 24000;
