import { type ToolDefinition, tool } from "@opencode-ai/plugin";
import { DREAMER_AGENT } from "../../agents/dreamer";
import { SIDEKICK_AGENT } from "../../agents/sidekick";
import {
    archiveMemory,
    CATEGORY_PRIORITY,
    getMemoriesByProject,
    getMemoryByHash,
    getMemoryById,
    insertMemoryIdempotent,
    type Memory,
    type MemoryCategory,
    mergeMemoryStats,
    saveEmbeddingIfHashMatches,
    supersededMemory,
    updateMemorySeenCount,
    V2_MEMORY_CATEGORIES,
} from "../../features/magic-context/memory";
import {
    embedTextForProject,
    enqueueShadowEmbeddingItems,
    getProjectEmbeddingSnapshot,
} from "../../features/magic-context/memory/embedding";
import { invalidateMemory } from "../../features/magic-context/memory/embedding-cache";
import { computeNormalizedHash } from "../../features/magic-context/memory/normalize-hash";
import {
    hasMemoryClassifiedAtColumn,
    hasMemoryShareableColumn,
} from "../../features/magic-context/memory/storage-memory";
import {
    normalizeStoredProjectPath,
    queueMemoryMutation,
    storedPathBelongsToIdentity,
} from "../../features/magic-context/storage";
import {
    expandWorkspaceIdentitySetWithAliases,
    resolveStoredPathWorkspaceIdentity,
    resolveWorkspaceIdentitySet,
    resolveWorkspaceShareCategories,
    storedPathBelongsToWorkspace,
} from "../../features/magic-context/workspaces";
import { sessionLog } from "../../shared/logger";
import {
    CTX_MEMORY_DESCRIPTION,
    CTX_MEMORY_TOOL_NAME,
    DEFAULT_LIST_LIMIT,
    DEFAULT_SEARCH_LIMIT,
} from "./constants";
import { formatMemoryList } from "./format-memory-list";
import {
    CTX_MEMORY_ACTIONS,
    CTX_MEMORY_DREAMER_ACTIONS,
    type CtxMemoryAction,
    type CtxMemoryArgs,
    type CtxMemoryToolDeps,
} from "./types";
import { runImmediateTransaction } from "./verification-recording";

const MEMORY_CATEGORIES = new Set<string>(CATEGORY_PRIORITY);

function isMemoryCategory(value: string): value is MemoryCategory {
    return MEMORY_CATEGORIES.has(value);
}

function normalizeLimit(limit?: number, fallback: number = DEFAULT_SEARCH_LIMIT): number {
    if (typeof limit !== "number" || !Number.isFinite(limit)) {
        return fallback;
    }

    return Math.max(1, Math.floor(limit));
}

function normalizeOffset(offset?: number): number {
    if (typeof offset !== "number" || !Number.isFinite(offset)) {
        return 0;
    }

    return Math.max(0, Math.floor(offset));
}

// When a caller omits `allowedActions`, fall back
// to the least-privileged set instead of the dreamer's full action list. The
// only production caller (`tool-registry.ts`) passes the primary set
// (`CTX_MEMORY_ACTIONS`) explicitly, and dreamer child sessions are gated by the
// runtime `toolContext.agent === DREAMER_AGENT` check below — they bypass
// `allowedActions` entirely. A future caller that forgets the field would
// previously have inadvertently let primary agents run the dreamer-only `list`;
// fail-closed default prevents that class of regression.
function getAllowedActions(deps: CtxMemoryToolDeps): [CtxMemoryAction, ...CtxMemoryAction[]] {
    const allowed = deps.allowedActions?.length ? deps.allowedActions : CTX_MEMORY_ACTIONS;
    return [...allowed] as [CtxMemoryAction, ...CtxMemoryAction[]];
}

function normalizeCategory(category?: string): string | undefined {
    const trimmed = category?.trim();
    return trimmed ? trimmed : undefined;
}

function filterByCategory(memories: Memory[], category?: string): Memory[] {
    if (!category) {
        return memories;
    }

    return memories.filter((memory) => memory.category === category);
}

function queueMemoryEmbedding(args: {
    deps: CtxMemoryToolDeps;
    sessionId: string;
    projectPath: string;
    memoryId: number;
    content: string;
}): void {
    const snapshot = getProjectEmbeddingSnapshot(args.projectPath);
    if (!snapshot?.enabled) {
        return;
    }

    const normalizedHash = computeNormalizedHash(args.content);
    void (async () => {
        const result = await embedTextForProject(args.projectPath, args.content);
        if (!result) {
            sessionLog(
                args.sessionId,
                `memory embedding skipped for memory ${args.memoryId}: provider unavailable or embedding generation failed.`,
            );
            return;
        }

        const saved = saveEmbeddingIfHashMatches(
            args.deps.db,
            args.memoryId,
            result.vector,
            result.modelId,
            normalizedHash,
        );
        if (!saved) {
            sessionLog(
                args.sessionId,
                `memory embedding skipped for memory ${args.memoryId}: content changed before the embedding finished.`,
            );
            return;
        }

        enqueueShadowEmbeddingItems(args.projectPath, "memory", [String(args.memoryId)]);
        sessionLog(args.sessionId, `proactively embedded memory ${args.memoryId}.`);
    })().catch((error: unknown) => {
        sessionLog(args.sessionId, `memory embedding failed for memory ${args.memoryId}:`, error);
    });
}

function getValidatedCategory(category: string | undefined): MemoryCategory | null {
    const trimmedCategory = category?.trim();

    if (!trimmedCategory) {
        return null;
    }

    if (!isMemoryCategory(trimmedCategory)) {
        return null;
    }

    return trimmedCategory;
}

function getDisabledMessage(): string {
    return "Cross-session memory is disabled for this project.";
}

function getSourceType(deps: CtxMemoryToolDeps) {
    return deps.sourceType ?? "agent";
}

function requestRustMemorySync(deps: CtxMemoryToolDeps, sessionId: string): void {
    try {
        deps.rustToolBackends?.memorySync?.(sessionId);
    } catch (error) {
        sessionLog(sessionId, "rust memory sync trigger failed (ignored):", error);
    }
}

interface MemoryProjectPathRow {
    project_path: string;
}

function projectPathForMemoryId(db: CtxMemoryToolDeps["db"], id: number): string | null {
    const row = db.prepare("SELECT project_path FROM memories WHERE id = ?").get(id) as
        | MemoryProjectPathRow
        | undefined;
    return row?.project_path ?? null;
}

function projectIdentityForStoredPath(rawProjectPath: string): string {
    return normalizeStoredProjectPath(rawProjectPath);
}

function memoryBelongsToProject(memory: Memory, projectPath: string): boolean {
    return storedPathBelongsToIdentity(memory.projectPath, projectPath);
}

function isPrimaryMutableMemory(memory: Memory): boolean {
    return (
        (memory.status === "active" || memory.status === "permanent") &&
        memory.supersededByMemoryId === null
    );
}

function inactiveMemoryError(id: number, action: "updating" | "merging" | "archiving"): string {
    return `Error: Memory with ID ${id} is archived or superseded; restore it before ${action}.`;
}

function updateMemoryContentInCurrentTransaction(
    db: CtxMemoryToolDeps["db"],
    memory: Memory,
    content: string,
    normalizedHash: string,
): void {
    db.prepare(
        "UPDATE memories SET content = ?, normalized_hash = ?, updated_at = ? WHERE id = ?",
    ).run(content, normalizedHash, Date.now(), memory.id);
    // The classify `shareable` verdict was scored against the OLD content; new
    // content invalidates it. Fail closed → private; the dreamer re-scores later.
    if (hasMemoryShareableColumn(db)) {
        db.prepare("UPDATE memories SET shareable = 0 WHERE id = ?").run(memory.id);
    }
    // Clear the classify marker so the changed fact is re-scored on the next
    // classify run (importance/scope were judged against the old content).
    if (hasMemoryClassifiedAtColumn(db)) {
        db.prepare("UPDATE memories SET classified_at = NULL WHERE id = ?").run(memory.id);
    }
    db.prepare("DELETE FROM memory_embeddings WHERE memory_id = ?").run(memory.id);
    invalidateMemory(memory.projectPath, memory.id);
}

function createCtxMemoryTool(deps: CtxMemoryToolDeps): ToolDefinition {
    const allowedActions = getAllowedActions(deps);

    return tool({
        description: CTX_MEMORY_DESCRIPTION,
        args: {
            // The OpenCode plugin exposes one shared tool definition for all agents, so
            // schema-level narrowing to `allowedActions` blocks dreamer child sessions
            // before execute() can inspect `toolContext.agent`. Keep the full action
            // schema visible to the runtime and enforce primary-session safety below.
            action: tool.schema
                .enum([...CTX_MEMORY_DREAMER_ACTIONS])
                .describe("What to do: write, update, archive, merge, or list"),
            content: tool.schema
                .string()
                .optional()
                .describe(
                    "The memory text — one standalone fact (required for write, update, merge)",
                ),
            category: tool.schema
                .enum([...V2_MEMORY_CATEGORIES])
                .optional()
                .describe(
                    "What kind of fact this is (required for write; optional merge override)",
                ),
            ids: tool.schema
                .array(tool.schema.number())
                .optional()
                .describe(
                    "Target memory id(s) from <project-memory>: update takes exactly one, archive one or more, merge two or more",
                ),
            limit: tool.schema
                .number()
                .optional()
                .describe("Maximum results to return for list (default: 100)"),
            offset: tool.schema
                .number()
                .optional()
                .describe("Zero-based offset for list pagination (default: 0)"),
            reason: tool.schema
                .string()
                .optional()
                .describe("Why the memory is being archived (optional, recommended)"),
        },
        async execute(args: CtxMemoryArgs, toolContext) {
            // Sidekick consumes untrusted `/ctx-aug` prompt text and is retrieval-only;
            // fail closed even if a future permission list accidentally exposes this tool.
            if (toolContext.agent === SIDEKICK_AGENT) {
                return "Error: ctx_memory is not available to the sidekick agent.";
            }
            if (toolContext.agent !== DREAMER_AGENT && !allowedActions.includes(args.action)) {
                return `Error: Action '${args.action}' is not allowed in this context.`;
            }

            // Resolve the session's actual project from `toolContext.directory`
            // each call. OpenCode's top-level `ctx.directory` (the launch dir)
            // can differ from the session's working directory when the user
            // runs `opencode -s <id>` from outside the project.
            const projectPath = deps.resolveProjectPath(toolContext.directory);
            await deps.ensureProjectRegistered?.(toolContext.directory, deps.db);
            const workspaceIdentitySet = resolveWorkspaceIdentitySet(deps.db, projectPath);
            const expandedWorkspace = expandWorkspaceIdentitySetWithAliases(
                deps.db,
                workspaceIdentitySet.identities,
            );
            const workspaceVisibleIdentities =
                workspaceIdentitySet.identities.length > 1
                    ? expandedWorkspace.expandedIdentities
                    : workspaceIdentitySet.identities;
            const targetIdentityForStoredPath = (rawProjectPath: string) =>
                workspaceIdentitySet.identities.length > 1
                    ? (resolveStoredPathWorkspaceIdentity(
                          rawProjectPath,
                          workspaceIdentitySet.identities,
                          expandedWorkspace.canonicalIdentityByStoredPath,
                      ) ?? projectIdentityForStoredPath(rawProjectPath))
                    : projectIdentityForStoredPath(rawProjectPath);
            // The workspace's share-category policy matches the render path.
            // null means there is no workspace filter; a workspaced caller gets
            // an explicit list where [] shares no foreign categories.
            const toolShareCategories =
                workspaceIdentitySet.identities.length > 1
                    ? resolveWorkspaceShareCategories(deps.db, projectPath)
                    : null;
            // Visibility is the READ contract: own memories are visible in every
            // category, while foreign workspace memories are visible only in
            // categories the workspace explicitly shares. Mutations by primary
            // agents use memoryOwnedByTool below so shared visibility never
            // grants write access to another project.
            const memoryVisibleToTool = (memory: Memory): boolean => {
                if (workspaceIdentitySet.identities.length <= 1) {
                    return memoryBelongsToProject(memory, projectPath);
                }
                if (
                    !storedPathBelongsToWorkspace(
                        memory.projectPath,
                        workspaceIdentitySet.identities,
                        workspaceVisibleIdentities,
                        expandedWorkspace.canonicalIdentityByStoredPath,
                    )
                ) {
                    return false;
                }
                const isOwn = targetIdentityForStoredPath(memory.projectPath) === projectPath;
                if (isOwn) return true;
                return toolShareCategories?.includes(memory.category) ?? false;
            };
            const memoryOwnedByTool = (memory: Memory): boolean =>
                workspaceIdentitySet.identities.length > 1
                    ? targetIdentityForStoredPath(memory.projectPath) === projectPath
                    : memoryBelongsToProject(memory, projectPath);
            const embeddingSnapshot = getProjectEmbeddingSnapshot(projectPath);
            if (
                embeddingSnapshot
                    ? !embeddingSnapshot.features.memoryEnabled
                    : deps.memoryEnabled === false
            ) {
                return getDisabledMessage();
            }

            if (args.action === "write") {
                const content = args.content?.trim();
                if (!content) {
                    return "Error: 'content' is required when action is 'write'.";
                }

                const rawCategory = args.category?.trim();
                if (!rawCategory) {
                    return "Error: 'category' is required when action is 'write'.";
                }

                const category = getValidatedCategory(rawCategory);
                if (!category) {
                    return `Error: Unknown memory category '${rawCategory}'.`;
                }

                const existingMemory = getMemoryByHash(
                    deps.db,
                    projectPath,
                    category,
                    computeNormalizedHash(content),
                );
                if (existingMemory) {
                    updateMemorySeenCount(deps.db, existingMemory.id);
                    requestRustMemorySync(deps, toolContext.sessionID);
                    return `Memory already exists [ID: ${existingMemory.id}] in ${category} (seen count incremented).`;
                }

                const insertResult = insertMemoryIdempotent(deps.db, {
                    projectPath: projectPath,
                    category,
                    content,
                    sourceSessionId: toolContext.sessionID,
                    sourceType:
                        toolContext.agent === DREAMER_AGENT ? "dreamer" : getSourceType(deps),
                });
                if (!insertResult.inserted) {
                    return `Memory already exists [ID: ${insertResult.memory.id}] in ${category} (seen count incremented).`;
                }

                queueMemoryEmbedding({
                    deps,
                    sessionId: toolContext.sessionID,
                    projectPath,
                    memoryId: insertResult.memory.id,
                    content,
                });
                requestRustMemorySync(deps, toolContext.sessionID);

                return `Saved memory [ID: ${insertResult.memory.id}] in ${category}.`;
            }

            if (args.action === "list") {
                const limit = normalizeLimit(args.limit, DEFAULT_LIST_LIMIT);
                const offset = normalizeOffset(args.offset);
                const category = normalizeCategory(args.category);
                const filtered = filterByCategory(
                    getMemoriesByProject(deps.db, projectPath),
                    category,
                );
                const pageMemories = filtered.slice(offset, offset + limit);

                return formatMemoryList({
                    pageMemories,
                    totalCount: filtered.length,
                    offset,
                    category,
                });
            }

            if (args.action === "update") {
                const updateIds = args.ids;
                if (updateIds?.length !== 1 || !updateIds.every(Number.isInteger)) {
                    return "Error: 'ids' must contain exactly one integer memory ID when action is 'update'.";
                }
                const updateId = updateIds[0];

                const content = args.content?.trim();
                if (!content) {
                    return "Error: 'content' is required when action is 'update'.";
                }

                const rawProjectPath = projectPathForMemoryId(deps.db, updateId);
                const memory = getMemoryById(deps.db, updateId);
                const updateAllowed = memory
                    ? toolContext.agent === DREAMER_AGENT
                        ? memoryVisibleToTool(memory)
                        : memoryOwnedByTool(memory)
                    : false;
                if (!memory || !rawProjectPath || !updateAllowed) {
                    return `Error: Memory with ID ${updateId} was not found.`;
                }
                if (toolContext.agent !== DREAMER_AGENT && !isPrimaryMutableMemory(memory)) {
                    return inactiveMemoryError(updateId, "updating");
                }

                const normalizedHash = computeNormalizedHash(content);
                const duplicate = getMemoryByHash(
                    deps.db,
                    targetIdentityForStoredPath(rawProjectPath),
                    memory.category,
                    normalizedHash,
                );
                if (duplicate && duplicate.id !== memory.id) {
                    return `Error: Memory content already exists as ID ${duplicate.id}; merge or archive duplicates instead.`;
                }

                const projectIdentity = targetIdentityForStoredPath(rawProjectPath);
                runImmediateTransaction(deps.db, () => {
                    updateMemoryContentInCurrentTransaction(
                        deps.db,
                        memory,
                        content,
                        normalizedHash,
                    );
                    queueMemoryMutation(deps.db, {
                        projectPath: projectIdentity,
                        mutationType: "update",
                        targetMemoryId: memory.id,
                        category: memory.category,
                        newContent: content,
                    });
                });
                queueMemoryEmbedding({
                    deps,
                    sessionId: toolContext.sessionID,
                    projectPath: projectIdentity,
                    memoryId: memory.id,
                    content,
                });
                requestRustMemorySync(deps, toolContext.sessionID);

                return `Updated memory [ID: ${memory.id}] in ${memory.category}.`;
            }

            if (args.action === "merge") {
                const ids = args.ids;
                if (!ids || ids.length < 2 || !ids.every(Number.isInteger)) {
                    return "Error: 'ids' must include at least two integer memory IDs when action is 'merge'.";
                }
                if (new Set(ids).size !== ids.length) {
                    return "Error: 'ids' must include at least two distinct memory IDs when action is 'merge'.";
                }

                const content = args.content?.trim();
                if (!content) {
                    return "Error: 'content' is required when action is 'merge'.";
                }

                const sourceMemories = ids
                    .map((id) => getMemoryById(deps.db, id))
                    .filter((memory): memory is Memory => Boolean(memory));
                if (sourceMemories.length !== ids.length) {
                    return "Error: One or more source memories were not found.";
                }
                // Cross-identity consolidation is a DREAMER-ONLY capability: the
                // loop below supersedes each source under ITS OWN project identity
                // and queues a per-project supersede-delta row, so every affected
                // project's m[1] reconciles. But `merge` is now in the primary
                // action set too, and a primary agent must not be able to reach
                // into ANOTHER project's memories. So mirror update/archive: a
                // non-dreamer caller may only merge memories that all belong to
                // its own resolved project. The dreamer keeps the cross-identity
                // path (see the "merging across identities" test).
                if (toolContext.agent !== DREAMER_AGENT) {
                    const foreign = sourceMemories.find((memory) => !memoryOwnedByTool(memory));
                    if (foreign) {
                        return `Error: Memory with ID ${foreign.id} was not found.`;
                    }
                    const inactive = sourceMemories.find(
                        (memory) => !isPrimaryMutableMemory(memory),
                    );
                    if (inactive) {
                        return inactiveMemoryError(inactive.id, "merging");
                    }
                } else if (workspaceIdentitySet.identities.length > 1) {
                    // The dreamer keeps its cross-PROJECT merge power (#5971) OUTSIDE
                    // a workspace (the branch above leaves non-workspace dreamer
                    // merges unrestricted). But INSIDE a workspace, per-category
                    // sharing is the user's explicit privacy boundary that even the
                    // system's own consolidation worker honors: a FOREIGN member's
                    // memory in a non-shared category (or a non-member project's
                    // memory) is off-limits. memoryVisibleToTool already encodes
                    // exactly that for the workspace case (own → true,
                    // foreign-shared-category → true, else → false).
                    const blocked = sourceMemories.find((memory) => !memoryVisibleToTool(memory));
                    if (blocked) {
                        return `Error: Memory with ID ${blocked.id} is in a category not shared with this workspace member and cannot be merged.`;
                    }
                }

                // A fact has exactly one category. If sources span categories they
                // are NOT genuine duplicates — one is miscategorized; archive the
                // redundant one instead. Merging across categories silently destroys
                // a distinct fact, so reject it structurally (not a prompt rule).
                const sourceCategories = new Set(sourceMemories.map((memory) => memory.category));
                if (sourceCategories.size > 1) {
                    return `Error: Cannot merge memories from different categories (${[...sourceCategories].join(", ")}). If they are genuine duplicates, one is miscategorized — archive the redundant one instead of merging across categories.`;
                }

                const category =
                    getValidatedCategory(args.category) ?? sourceMemories[0]?.category ?? null;
                if (!category) {
                    return "Error: A valid category is required when action is 'merge'.";
                }

                const normalizedHash = computeNormalizedHash(content);

                const mergedFrom = JSON.stringify(
                    Array.from(
                        new Set(
                            sourceMemories.flatMap((memory) => {
                                let parsed: unknown[];
                                try {
                                    parsed = memory.mergedFrom ? JSON.parse(memory.mergedFrom) : [];
                                } catch {
                                    parsed = [];
                                }
                                return [
                                    memory.id,
                                    ...(Array.isArray(parsed)
                                        ? parsed.filter(
                                              (value): value is number => typeof value === "number",
                                          )
                                        : []),
                                ];
                            }),
                        ),
                    ).sort((left, right) => left - right),
                );
                const mergedSeenCount = sourceMemories.reduce(
                    (sum, memory) => sum + memory.seenCount,
                    0,
                );
                const mergedRetrievalCount = sourceMemories.reduce(
                    (sum, memory) => sum + memory.retrievalCount,
                    0,
                );
                const mergedStatus = sourceMemories.some((memory) => memory.status === "permanent")
                    ? "permanent"
                    : "active";

                let mergeConflict: string | null = null;
                const canonicalMemory = runImmediateTransaction(deps.db, () => {
                    const lockedDuplicate = getMemoryByHash(
                        deps.db,
                        projectPath,
                        category,
                        normalizedHash,
                    );
                    const canonicalExisting =
                        lockedDuplicate && ids.includes(lockedDuplicate.id)
                            ? lockedDuplicate
                            : null;
                    if (lockedDuplicate && !canonicalExisting) {
                        mergeConflict = `Error: Memory content already exists as ID ${lockedDuplicate.id}; update or archive existing duplicates instead.`;
                        return null;
                    }

                    const nextCanonical =
                        canonicalExisting?.id != null
                            ? canonicalExisting
                            : insertMemoryIdempotent(deps.db, {
                                  projectPath: projectPath,
                                  category,
                                  content,
                                  sourceSessionId: toolContext.sessionID,
                                  sourceType:
                                      toolContext.agent === DREAMER_AGENT
                                          ? "dreamer"
                                          : getSourceType(deps),
                              }).memory;
                    const canonicalContentChanged =
                        nextCanonical.content !== content ||
                        nextCanonical.normalizedHash !== normalizedHash;

                    if (canonicalContentChanged) {
                        updateMemoryContentInCurrentTransaction(
                            deps.db,
                            nextCanonical,
                            content,
                            normalizedHash,
                        );
                    }

                    mergeMemoryStats(
                        deps.db,
                        nextCanonical.id,
                        mergedSeenCount,
                        mergedRetrievalCount,
                        mergedFrom,
                        mergedStatus,
                    );

                    for (const memory of sourceMemories) {
                        if (memory.id === nextCanonical.id) {
                            continue;
                        }
                        supersededMemory(deps.db, memory.id, nextCanonical.id);
                        queueMemoryMutation(deps.db, {
                            projectPath: projectIdentityForStoredPath(memory.projectPath),
                            mutationType: "superseded",
                            targetMemoryId: memory.id,
                            supersededById: nextCanonical.id,
                        });
                    }

                    if (canonicalExisting && canonicalContentChanged) {
                        queueMemoryMutation(deps.db, {
                            projectPath: projectIdentityForStoredPath(nextCanonical.projectPath),
                            mutationType: "update",
                            targetMemoryId: nextCanonical.id,
                            category,
                            newContent: content,
                        });
                    }

                    return nextCanonical;
                });
                if (mergeConflict || !canonicalMemory) {
                    return mergeConflict ?? "Error: Failed to merge memories.";
                }

                queueMemoryEmbedding({
                    deps,
                    sessionId: toolContext.sessionID,
                    projectPath,
                    memoryId: canonicalMemory.id,
                    content,
                });
                requestRustMemorySync(deps, toolContext.sessionID);

                const supersededIds = sourceMemories
                    .map((memory) => memory.id)
                    .filter((id) => id !== canonicalMemory.id);
                return `Merged memories [${ids.join(", ")}] into canonical memory [ID: ${canonicalMemory.id}] in ${category}; superseded [${supersededIds.join(", ")}].`;
            }

            if (args.action === "archive") {
                const rawArchiveIds = args.ids;
                if (
                    !rawArchiveIds ||
                    rawArchiveIds.length === 0 ||
                    !rawArchiveIds.every(Number.isInteger)
                ) {
                    return "Error: 'ids' must contain at least one integer memory ID when action is 'archive'.";
                }
                // De-dupe (first-seen order) so `ids:[42,42]` archives once and
                // queues one mutation-log row instead of two.
                const archiveIds = [...new Set(rawArchiveIds)];

                // Validate the whole batch BEFORE mutating anything so a typo'd
                // id can't half-archive a batch (all-or-nothing, matching the
                // single-transaction write below).
                const targets: Array<{ memoryId: number; projectIdentity: string }> = [];
                for (const memoryId of archiveIds) {
                    const rawProjectPath = projectPathForMemoryId(deps.db, memoryId);
                    const memory = getMemoryById(deps.db, memoryId);
                    const archiveAllowed = memory
                        ? toolContext.agent === DREAMER_AGENT
                            ? memoryVisibleToTool(memory)
                            : memoryOwnedByTool(memory)
                        : false;
                    if (!memory || !rawProjectPath || !archiveAllowed) {
                        return `Error: Memory with ID ${memoryId} was not found.`;
                    }
                    if (toolContext.agent !== DREAMER_AGENT && !isPrimaryMutableMemory(memory)) {
                        // Mirror update/merge: once the primary agent archived or
                        // superseded this memory, re-archiving it should return the
                        // same friendly inactive-memory error instead of mutating it.
                        return inactiveMemoryError(memoryId, "archiving");
                    }
                    targets.push({
                        memoryId,
                        projectIdentity: targetIdentityForStoredPath(rawProjectPath),
                    });
                }

                runImmediateTransaction(deps.db, () => {
                    for (const target of targets) {
                        archiveMemory(deps.db, target.memoryId, args.reason);
                        queueMemoryMutation(deps.db, {
                            projectPath: target.projectIdentity,
                            mutationType: "archive",
                            targetMemoryId: target.memoryId,
                        });
                    }
                });
                requestRustMemorySync(deps, toolContext.sessionID);
                const idList = targets.map((t) => t.memoryId).join(", ");
                const plural = targets.length > 1 ? "memories" : "memory";
                return args.reason?.trim()
                    ? `Archived ${plural} [ID: ${idList}] (${args.reason.trim()}).`
                    : `Archived ${plural} [ID: ${idList}].`;
            }

            return "Error: Unknown action.";
        },
    });
}

export function createCtxMemoryTools(deps: CtxMemoryToolDeps): Record<string, ToolDefinition> {
    return {
        [CTX_MEMORY_TOOL_NAME]: createCtxMemoryTool(deps),
    };
}
