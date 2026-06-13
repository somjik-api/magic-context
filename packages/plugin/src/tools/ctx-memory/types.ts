import type { MemorySourceType } from "../../features/magic-context/memory";
import type { Database } from "../../shared/sqlite";

// Actions a PRIMARY (non-dreamer) agent may run. Primary agents see active
// memories — with their ids — in the injected <project-memory> block, so they
// can target a specific memory to archive/update/merge in-session without
// waiting for the dreamer. `archive` is the single soft-remove action (sets
// status='archived'); the former `delete` action was an exact alias of it and
// was removed. Only `list` (bulk enumeration — large output, and unnecessary
// since active memories are already in context) stays dreamer-only.
export const CTX_MEMORY_ACTIONS = ["write", "archive", "update", "merge"] as const;

export const CTX_MEMORY_DREAMER_ACTIONS = [...CTX_MEMORY_ACTIONS, "list"] as const;

export type CtxMemoryAction = (typeof CTX_MEMORY_DREAMER_ACTIONS)[number];

export interface CtxMemoryArgs {
    action: CtxMemoryAction;
    content?: string;
    category?: string;
    /**
     * Target memory id(s). One unified parameter for all id-taking actions:
     * update requires exactly one, archive one or more (batch), merge two or
     * more. The former scalar `id` param was folded in here.
     */
    ids?: number[];
    limit?: number;
    offset?: number;
    reason?: string;
}

export interface CtxMemoryToolDeps {
    db: Database;
    ensureProjectRegistered?: (directory: string, db: Database) => Promise<void>;
    /**
     * Resolve the project identity for the active session's directory.
     *
     * Why a function instead of a baked string: OpenCode's top-level
     * `ctx.directory` is the directory the OpenCode process was started
     * in (often `$HOME` when launched via `opencode -s <id>` from outside
     * the project). The session's actual working directory is exposed
     * per-call via `toolContext.directory`. Resolving here ensures
     * `ctx_memory` operates on the session's project, not the launch
     * directory's project.
     */
    resolveProjectPath: (directory: string) => string;
    memoryEnabled?: boolean;
    embeddingEnabled?: boolean;
    allowedActions?: CtxMemoryAction[];
    sourceType?: MemorySourceType;
}
