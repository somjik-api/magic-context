import { Buffer } from "node:buffer";
import {
    buildCompartmentBlock,
    type Compartment,
    type CompartmentDateRanges,
    escapeXmlAttr,
    escapeXmlContent,
    getCompartments,
    getLastCompartmentEndMessageId,
    type SessionFact,
} from "../../features/magic-context/compartment-storage";
import {
    CATEGORY_PRIORITY,
    MEMORY_CATEGORY_ORDER_PRIORITY,
    MEMORY_CATEGORY_ORDER_SQL,
    MEMORY_CATEGORY_ORDER_UNKNOWN,
} from "../../features/magic-context/memory/constants";
import {
    getMaxMemoryIdForProjects,
    getMemoriesByProject,
    getMemoriesByProjects,
    getMemorySelectColumns,
    isMemoryRow,
    readNewMemoriesForM1Union,
} from "../../features/magic-context/memory/storage-memory";
import type { Memory, MemoryCategory } from "../../features/magic-context/memory/types";
import {
    computeProjectDocsHash,
    GLOBAL_USER_PROFILE_PROJECT_PATH,
    getMaxM0MutationId,
    getMaxMemoryMutationId,
    getMaxMemoryMutationIdForProjects,
    getMemoryMutationsForRender,
    getMemoryMutationsForRenderByProjects,
    getProjectState,
    persistCachedM0,
    readProjectDocsCanonical,
} from "../../features/magic-context/storage";
import {
    getActiveUserMemories,
    type UserMemory,
} from "../../features/magic-context/user-memory/storage-user-memory";
import {
    computeWorkspaceEpochFingerprint,
    expandWorkspaceIdentitySetWithAliases,
    resolveStoredPathWorkspaceIdentity,
    resolveWorkspaceIdentitySet,
    resolveWorkspaceShareCategories,
    sourceNameForMemory,
    type WorkspaceIdentitySet,
} from "../../features/magic-context/workspaces";
import { BoundedSessionMap } from "../../shared/bounded-session-map";
import { sessionLog } from "../../shared/logger";
import type { Database, Statement as PreparedStatement } from "../../shared/sqlite";
import { extractM0Block, renderCompartmentAtTier, renderDecayedCompartments } from "./decay-render";
import { buildKeyFilesBlock, type KeyFilesConfigForRender } from "./key-files-block";
import { getMessageTimesFromOpenCodeDb } from "./read-session-db";
import { estimateTokens } from "./read-session-formatting";
import type { MessageLike } from "./tag-messages";
import { formatDate } from "./temporal-awareness";

export interface PreparedCompartmentInjection {
    block: string;
    compartmentEndMessage: number;
    compartmentEndMessageId: string | null;
    compartmentCount: number;
    skippedVisibleMessages: number;
    factCount: number;
    memoryCount: number;
    rebuiltFromDb: boolean;
}

/**
 * In-memory cache of the last compartment injection result per session.
 * On non-flush passes, the cached result is replayed so that historian
 * publications between passes do not bust the Anthropic prompt-cache prefix.
 * The cache is invalidated explicitly via clearInjectionCache() after
 * historian/compressor/recomp write new compartments or facts.
 *
 * Bounded LRU: session.deleted clears entries explicitly, but sessions that
 * are never deleted (crashed OpenCode, force-quit, archived sessions) would
 * otherwise leak PreparedCompartmentInjection objects holding tens of KB of
 * XML each. 100 is generously above any realistic working set of active
 * sessions — evicted entries are simply recomputed on the next cache-busting
 * pass from the authoritative SQLite compartment state.
 */
const INJECTION_CACHE_MAX = 100;
type InjectionCacheEntry =
    | { kind: "empty"; compartmentEndMessageId: string; renderedBytes: number }
    | { kind: "populated"; injection: PreparedCompartmentInjection };

const injectionCache = new BoundedSessionMap<InjectionCacheEntry>(INJECTION_CACHE_MAX);

export function clearInjectionCache(sessionId: string): void {
    injectionCache.delete(sessionId);
}

/**
 * Return the set of memory ids currently rendered in the cached
 * <session-history> block for this session, if any. Used by ctx_search
 * to hard-filter memories the agent already sees in context — retrieving
 * them from search wastes tokens and pushes high-signal raw-history hits
 * further down the ranking.
 *
 * Returns null when no cache exists or the JSON payload is malformed
 * (callers should treat null as "don't filter" — the worst case is a
 * redundant memory result, not a correctness issue).
 */
export function getVisibleMemoryIds(db: Database, sessionId: string): Set<number> | null {
    try {
        const row = db
            .prepare("SELECT memory_block_ids FROM session_meta WHERE session_id = ?")
            .get(sessionId) as { memory_block_ids: string | null } | null;
        if (!row?.memory_block_ids) return null;
        const parsed = JSON.parse(row.memory_block_ids) as unknown;
        if (!Array.isArray(parsed)) return null;
        const ids = new Set<number>();
        for (const value of parsed) {
            if (typeof value === "number" && Number.isFinite(value)) {
                ids.add(value);
            }
        }
        return ids.size > 0 ? ids : null;
    } catch {
        return null;
    }
}

export interface CompartmentInjectionResult {
    injected: boolean;
    compartmentEndMessage: number;
    compartmentCount: number;
    skippedVisibleMessages: number;
}

export function renderMemoryBlock(memories: Memory[]): string | null {
    const byCategory = new Map<MemoryCategory, Memory[]>();
    for (const m of memories) {
        const existing = byCategory.get(m.category);
        if (existing) {
            existing.push(m);
        } else {
            byCategory.set(m.category, [m]);
        }
    }

    const sections: string[] = [];
    for (const category of CATEGORY_PRIORITY) {
        const categoryMemories = byCategory.get(category);
        if (!categoryMemories || categoryMemories.length === 0) {
            continue;
        }
        sections.push(
            `<${category}>`,
            ...categoryMemories.map((m) => `- ${escapeXmlContent(m.content)}`),
            `</${category}>`,
        );
    }

    if (sections.length === 0) {
        return null;
    }

    return `<project-memory>\n${sections.join("\n")}\n</project-memory>`;
}

/** Constraint keywords that signal a memory encodes a rule rather than a description. */
const CONSTRAINT_KEYWORDS = /\b(must|never|always|cannot|should not|must not)\b/i;

/**
 * Assign a utility tier to a memory for injection priority.
 * Lower tier = higher priority (packed first).
 *
 * Tier 0: Agent actually searched for and found this memory.
 * Tier 1: Contains constraint/rule keywords — likely guards against a real bug.
 * Tier 2: Everything else.
 */
function utilityTier(m: Memory): number {
    if (m.retrievalCount > 0) return 0;
    if (CONSTRAINT_KEYWORDS.test(m.content)) return 1;
    return 2;
}

/**
 * Sort memories by priority and trim to budget.
 *
 * Priority order:
 *   1. permanent status first
 *   2. utility tier (retrieved > constraint > other)
 *   3. seen count descending
 *   4. shorter content first (fit more memories in budget)
 *   5. deterministic id tiebreaker for cache stability
 *
 * Uses the real Claude tokenizer (via estimateTokens) so the trim stays
 * consistent with the rest of the plugin's token math — mismatching units
 * (chars/4 here vs real tokens elsewhere) caused either under- or
 * over-injection of memories, depending on memory content shape.
 */
export function trimMemoriesToBudget(
    sessionId: string,
    memories: Memory[],
    budgetTokens: number,
): Memory[] {
    const sorted = [...memories].sort((a, b) => {
        // Permanent memories first
        if (a.status === "permanent" && b.status !== "permanent") return -1;
        if (b.status === "permanent" && a.status !== "permanent") return 1;
        // Then by utility tier (lower = higher priority)
        const tierDiff = utilityTier(a) - utilityTier(b);
        if (tierDiff !== 0) return tierDiff;
        // Then by seen count descending (more frequently seen = higher priority)
        const seenDiff = b.seenCount - a.seenCount;
        if (seenDiff !== 0) return seenDiff;
        // Prefer shorter memories so more fit in budget
        const lenDiff = a.content.length - b.content.length;
        if (lenDiff !== 0) return lenDiff;
        // Deterministic tiebreaker by id to ensure stable ordering for cache safety
        return a.id - b.id;
    });

    const result: Memory[] = [];
    let usedTokens = 0;

    for (const memory of sorted) {
        // Estimate the rendered memory line ("- {content}") with the real
        // Claude tokenizer, plus a fixed ~6-token allowance for opening and
        // closing XML category tags amortized per item. Keeps units
        // consistent with rpc-handlers.ts / transform.ts / system-prompt-hash.ts
        // so the sidebar's "Memories" segment matches what actually lands in
        // the injection block.
        const memoryTokens = estimateTokens(`- ${memory.content}`) + 6;
        if (usedTokens + memoryTokens > budgetTokens) {
            break;
        }
        result.push(memory);
        usedTokens += memoryTokens;
    }

    if (result.length < memories.length) {
        sessionLog(
            sessionId,
            `trimmed memories from ${memories.length} to ${result.length} to fit injection budget of ${budgetTokens} tokens`,
        );
    }

    return result;
}

export function prepareCompartmentInjection(
    db: Database,
    sessionId: string,
    messages: MessageLike[],
    isCacheBusting: boolean,
    projectPath?: string,
    injectionBudgetTokens?: number,
    temporalAwareness?: boolean,
): PreparedCompartmentInjection | null {
    // On defer (cache-safe) passes, replay the cached injection result so that
    // historian publications between passes do not bust the prompt-cache prefix.
    const cached = injectionCache.get(sessionId);
    if (!isCacheBusting && cached) {
        if (cached.kind === "empty") {
            return null;
        }
        const prepared = cached.injection;
        if (prepared.compartmentEndMessageId === null) {
            sessionLog(
                sessionId,
                "compartment injection cache in degraded mode (null boundary), forcing rebuild",
            );
        } else {
            // Re-do the splice with the cached boundary (messages are rebuilt fresh each pass)
            if (prepared.compartmentEndMessageId.length > 0) {
                const cutoffIndex = messages.findIndex(
                    (message) => message.info.id === prepared.compartmentEndMessageId,
                );
                if (cutoffIndex >= 0) {
                    const remaining = messages.slice(cutoffIndex + 1);
                    messages.splice(0, messages.length, ...remaining);
                } else {
                    // Boundary message not in array — covered messages were already
                    // trimmed by OpenCode (compaction, old history not sent). The splice
                    // is effectively a no-op because there's nothing to splice out.
                    // Keep the cached injection so <session-history> stays stable on
                    // defer passes instead of alternating between injected/not-injected.
                    sessionLog(
                        sessionId,
                        `compartment injection: cached boundary ${prepared.compartmentEndMessageId} not in messages (already trimmed), reusing cache`,
                    );
                }
            }
            return { ...prepared, rebuiltFromDb: false };
        }
    }

    const compartments = getCompartments(db, sessionId);
    // v2 faithful facts: session_facts is retired as a render source. Facts are
    // promoted to project memory and render via <project-memory>. We no longer
    // read or render session_facts here (matching the runner's removed write
    // side); legacy pre-v2 rows are left un-rendered until /ctx-session-upgrade.
    const facts: SessionFact[] = [];

    let memoryBlock: string | undefined;
    let memoryCount = 0;
    if (projectPath) {
        // Use cached memory block to avoid cache busting on background changes (ctx_memory write, promotion).
        // Cache is cleared by replaceSessionFacts/replaceAllCompartmentState after historian/compressor/recomp.
        // Audit note: `as` cast is safe here — session_meta schema is owned by this plugin and the two
        // columns are guaranteed present after initializeDatabase(). A type guard would add overhead on a
        // hot path (every transform) for a table we fully control.
        const cachedMemory = db
            .prepare(
                "SELECT memory_block_cache, memory_block_count FROM session_meta WHERE session_id = ?",
            )
            .get(sessionId) as { memory_block_cache: string; memory_block_count: number } | null;

        if (cachedMemory?.memory_block_cache) {
            memoryBlock = cachedMemory.memory_block_cache;
            memoryCount = cachedMemory.memory_block_count;
        } else {
            let memories = getMemoriesByProject(db, projectPath, ["active", "permanent"]);
            if (injectionBudgetTokens && memories.length > 0) {
                memories = trimMemoriesToBudget(sessionId, memories, injectionBudgetTokens);
            }
            memoryCount = memories.length;
            memoryBlock = renderMemoryBlock(memories) ?? undefined;
            // Capture ids of memories actually rendered in the block. Stored in
            // session_meta.memory_block_ids as JSON so ctx_search can hard-filter
            // them out of search results (the agent already sees them in <session-history>).
            const renderedIds = memories.map((m) => m.id);

            // Snapshot so subsequent turns reuse the same block without cache bust.
            // Swallow SQLITE_BUSY: the cache is a pure optimization (the block itself
            // is already computed and returned below). If another writer holds the DB
            // past SQLite busy_timeout — typically a concurrent dreamer/historian child
            // session or a second OpenCode process — we'd rather let the transform
            // proceed with a one-turn cache miss than crash the user's prompt.
            // Issue: https://github.com/cortexkit/magic-context/issues/23
            try {
                db.prepare(
                    "UPDATE session_meta SET memory_block_cache = ?, memory_block_count = ?, memory_block_ids = ? WHERE session_id = ?",
                ).run(memoryBlock ?? "", memoryCount, JSON.stringify(renderedIds), sessionId);
            } catch (error) {
                const code = (error as { code?: string } | null)?.code;
                if (code === "SQLITE_BUSY") {
                    sessionLog(
                        sessionId,
                        "memory_block_cache UPDATE hit SQLITE_BUSY, skipping snapshot for this turn",
                    );
                } else {
                    throw error;
                }
            }
        }
    }

    // Nothing to inject if we have no compartments, no facts, and no memories
    if (compartments.length === 0 && facts.length === 0 && !memoryBlock) {
        injectionCache.set(sessionId, {
            kind: "empty",
            compartmentEndMessageId: "",
            renderedBytes: 0,
        });
        return null;
    }

    let dateRanges: CompartmentDateRanges | undefined;
    if (temporalAwareness && compartments.length > 0) {
        // Resolve start/end message times from OpenCode's DB in a single batched query.
        const ids = new Set<string>();
        for (const c of compartments) {
            if (c.startMessageId) ids.add(c.startMessageId);
            if (c.endMessageId) ids.add(c.endMessageId);
        }
        const times = getMessageTimesFromOpenCodeDb(sessionId, Array.from(ids));
        const byId = new Map<number, { start: string; end: string }>();
        for (const c of compartments) {
            const startMs = times.get(c.startMessageId);
            const endMs = times.get(c.endMessageId);
            if (startMs !== undefined && endMs !== undefined) {
                byId.set(c.id, { start: formatDate(startMs), end: formatDate(endMs) });
            }
        }
        if (byId.size > 0) dateRanges = { byId };
    }

    const block = buildCompartmentBlock(compartments, facts, memoryBlock, dateRanges);

    // When there are no compartments yet (new session, or memories seeded before
    // historian first run), inject memories/facts without a boundary cutoff.
    // No messages are spliced because there's nothing to replace — the block is
    // prepended to message[0] the same way system-level context is.
    if (compartments.length === 0) {
        const result: PreparedCompartmentInjection = {
            block,
            compartmentEndMessage: 0,
            compartmentEndMessageId: "",
            compartmentCount: 0,
            skippedVisibleMessages: 0,
            factCount: facts.length,
            memoryCount,
            rebuiltFromDb: true,
        };
        injectionCache.set(sessionId, { kind: "populated", injection: result });
        return result;
    }

    const lastCompartment = compartments[compartments.length - 1];
    const lastEnd = lastCompartment.endMessage;
    const lastEndMessageId = lastCompartment.endMessageId;

    // Trim boundary selection. On a CACHE-BUSTING pass, trim to the latest
    // compartment — m[1] will re-render to cover it. On a NON-cache-busting
    // (defer) pass that reaches this REBUILD path, the in-memory injection cache
    // was cold (a fresh process after a restart): the persisted m[0]/m[1] summary
    // is replayed stale, so a compartment published after the last
    // materialize/soft-refresh is summarized in NEITHER m[1] NOR m[0]. Trimming
    // to the latest boundary would also drop its raw messages → silent history
    // loss until the next exec pass. Instead trim only to the boundary the cached
    // summary actually covers (cached_m0_last_baseline_end_message_id), keeping
    // the newer compartment's raw messages in the live tail. That column is
    // written ONLY by the m0/m1 materialize/soft-refresh path, so its presence
    // self-gates this to v2 sessions; absent (legacy / never materialized) →
    // fall back to the latest boundary.
    let trimEndMessageId = lastEndMessageId;
    if (!isCacheBusting) {
        const baseline = readCachedBaselineState(db, sessionId);
        if (baseline.hasCachedM0) {
            // v2 cold defer rebuild (in-memory cache lost post-restart). Trim ONLY
            // to what the replayed cached m[1] actually covers.
            if (baseline.boundary) {
                trimEndMessageId = baseline.boundary;
            } else {
                // hasCachedM0 but null boundary: m[0]/m[1] was materialized BEFORE
                // any compartment boundary existed (the common new-session case — a
                // fresh session materializes m[0] with 0 compartments, then the
                // first historian publish lands, then a restart before the next
                // exec pass). The cached m[1] summarizes NONE of the current
                // compartments, so trimming to the latest boundary would drop a
                // compartment's raw messages that live in neither m[0] nor m[1] →
                // silent history loss. Suppress the trim entirely: keep all raw
                // messages in the tail; the next exec pass folds them into m[1].
                trimEndMessageId = "";
            }
        }
        // else: legacy / never-materialized v1 session (no cached m[0]) → keep the
        // latest-compartment boundary (the original v1 trim behavior).
    }

    if (trimEndMessageId.length === 0) {
        sessionLog(
            sessionId,
            "injecting legacy compartments without visible-prefix trimming because latest stored compartment has no end_message_id",
            {
                compartmentCount: compartments.length,
                compartmentEndMessage: lastEnd,
            },
        );
        const result: PreparedCompartmentInjection = {
            block,
            compartmentEndMessage: lastEnd,
            compartmentEndMessageId: "",
            compartmentCount: compartments.length,
            skippedVisibleMessages: 0,
            factCount: facts.length,
            memoryCount,
            rebuiltFromDb: true,
        };
        injectionCache.set(sessionId, { kind: "populated", injection: result });
        return result;
    }

    let skippedVisibleMessages = 0;
    const cutoffIndex = messages.findIndex((message) => message.info.id === trimEndMessageId);
    if (cutoffIndex >= 0) {
        skippedVisibleMessages = cutoffIndex + 1;
        const remaining = messages.slice(cutoffIndex + 1);
        messages.splice(0, messages.length, ...remaining);
    } else {
        sessionLog(
            sessionId,
            `compartment injection entering degraded mode: boundary ${trimEndMessageId} not in visible messages`,
        );
    }

    const result: PreparedCompartmentInjection = {
        block,
        compartmentEndMessage: lastEnd,
        compartmentEndMessageId: cutoffIndex >= 0 ? trimEndMessageId : null,
        compartmentCount: compartments.length,
        skippedVisibleMessages,
        factCount: facts.length,
        memoryCount,
        rebuiltFromDb: true,
    };
    injectionCache.set(sessionId, { kind: "populated", injection: result });
    return result;
}

/**
 * Read the persisted m[0]/m[1] baseline state for the cold-rebuild trim decision:
 *   - `hasCachedM0`: a v2 cached m[0] snapshot exists. Distinguishes a
 *     materialized-but-boundaryless session (null boundary is meaningful → the
 *     summary covers NO compartment, so do not trim) from a legacy /
 *     never-materialized v1 session (no cache → fall back to latest boundary).
 *   - `boundary`: the latest compartment end message id the cached m[1] covers,
 *     or null when m[0] was materialized before any compartment boundary existed.
 *
 * `hasCachedM0` is the discriminator, NOT boundary-nullness: null boundary with a
 * present cache is a legitimate state (a fresh session materializes m[0] with 0
 * compartments), and treating it as "fall back to latest" reintroduced the very
 * history-loss the cold-rebuild trim exists to prevent.
 */
function readCachedBaselineState(
    db: Database,
    sessionId: string,
): { hasCachedM0: boolean; boundary: string | null } {
    const row = db
        .prepare(
            "SELECT cached_m0_bytes AS m0, cached_m0_last_baseline_end_message_id AS boundary FROM session_meta WHERE session_id = ?",
        )
        .get(sessionId) as { m0: unknown; boundary: string | null } | undefined;
    const boundary = row?.boundary;
    return {
        hasCachedM0: row?.m0 != null,
        boundary: boundary && boundary.length > 0 ? boundary : null,
    };
}

export function renderCompartmentInjection(
    sessionId: string,
    messages: MessageLike[],
    prepared: PreparedCompartmentInjection,
): CompartmentInjectionResult {
    const historyBlock = `<session-history>\n${prepared.block}\n</session-history>`;
    const firstMessage = messages[0];
    const textPart = firstMessage ? findFirstTextPart(firstMessage.parts) : null;
    if (!firstMessage || !textPart || isDroppedPlaceholder(textPart.text)) {
        // synthetic: true — injected context, not a real user turn. Keeps it out
        // of OpenCode's auto-title gate (issue #129) while still reaching the
        // model (toModelMessagesEffect filters `ignored`, not `synthetic`).
        messages.unshift({
            info: { role: "user", sessionID: sessionId },
            parts: [{ type: "text", text: historyBlock, synthetic: true }],
        });
    } else {
        textPart.text = `${historyBlock}\n\n${textPart.text}`;
    }

    const memoryLabel = prepared.memoryCount > 0 ? ` + ${prepared.memoryCount} memories` : "";
    if (prepared.compartmentCount > 0) {
        sessionLog(
            sessionId,
            `injected ${prepared.compartmentCount} compartments + ${prepared.factCount} facts${memoryLabel} into message[0]`,
        );
    } else {
        sessionLog(
            sessionId,
            `injected ${prepared.factCount} facts${memoryLabel} into message[0] (no compartments yet)`,
        );
    }

    return {
        injected: true,
        compartmentEndMessage: prepared.compartmentEndMessage,
        compartmentCount: prepared.compartmentCount,
        skippedVisibleMessages: prepared.skippedVisibleMessages,
    };
}

function findFirstTextPart(parts: unknown[]): { type: string; text: string } | null {
    for (const part of parts) {
        if (part === null || typeof part !== "object") continue;
        const p = part as Record<string, unknown>;
        if (p.type === "text" && typeof p.text === "string" && !p.ignored) {
            return p as unknown as { type: string; text: string };
        }
    }
    return null;
}

function isDroppedPlaceholder(text: string): boolean {
    return /^\[dropped §\d+§\]$/.test(text.trim());
}

export interface M0SnapshotMarkers {
    projectMemoryEpoch: number;
    workspaceFingerprint: string | null;
    projectUserProfileVersion: number;
    maxCompartmentSeq: number;
    maxMemoryId: number;
    maxMutationId: number;
    maxMemoryMutationId: number;
    projectDocsHash: string;
    materializedAt: number;
    sessionFactsVersion: number;
    upgradeState: string | null;
    // HARD-bust markers: provider-side cache-eviction signals. A change in any
    // of these means the Anthropic prompt cache was already dead (tools/system
    // block changed, or model switched), so folding m[1] into m[0] is "free".
    // Captured from runtime signals at the injectM0M1 call site (NOT a pure DB
    // read), so readCurrentM0SnapshotMarkers takes them as inputs.
    systemHash: string;
    modelKey: string;
}

/**
 * Runtime cache-eviction signals threaded into the materialization decision.
 * These are NOT derived from durable DB state like the content markers — they
 * come from the current flight (system-prompt hash, tool-set fingerprint,
 * provider/model key) plus the TTL idle window.
 */
export interface M0HardSignals {
    systemHash: string;
    modelKey: string;
    /** True when the provider cache TTL has elapsed since lastResponseTime. */
    cacheExpired: boolean;
    /** Epoch ms of the last completed assistant response (end-of-turn). */
    lastResponseTime: number;
}

const EMPTY_HARD_SIGNALS: M0HardSignals = {
    systemHash: "",
    modelKey: "",
    cacheExpired: false,
    lastResponseTime: 0,
};

export interface M0M1State {
    sessionId: string;
    isSubagent?: boolean;
    cachedM0Bytes: Buffer | null;
    cachedM1Bytes: Buffer | null;
    cachedM0ProjectMemoryEpoch: number | null;
    cachedM0WorkspaceFingerprint: string | null;
    cachedM0ProjectUserProfileVersion: number | null;
    cachedM0MaxCompartmentSeq: number | null;
    cachedM0MaxMemoryId: number | null;
    cachedM0MaxMutationId: number | null;
    cachedM0MaxMemoryMutationId: number | null;
    cachedM0ProjectDocsHash: string | null;
    cachedM0MaterializedAt: number | null;
    cachedM0SessionFactsVersion: number | null;
    cachedM0UpgradeState: string | null;
    cachedM0SystemHash: string | null;
    cachedM0ToolSetHash: string | null;
    cachedM0ModelKey: string | null;
    snapshotMarkers?: M0SnapshotMarkers | null;
}

export interface M0M1RenderOptions {
    db: Database;
    sessionId: string;
    messages?: MessageLike[];
    state: M0M1State;
    projectPath?: string;
    projectDirectory?: string;
    memoryInjectionBudgetTokens?: number;
    historyBudgetTokens?: number;
    userProfileBudgetTokens?: number;
    keyFiles?: KeyFilesConfigForRender;
    isCacheBustingPass?: boolean;
    /** Provider-side cache-eviction signals for HARD-bust detection. */
    hardSignals?: M0HardSignals;
    preRenderedKeyFilesBlock?: string | null;
    workspaceIdentitySet?: WorkspaceIdentitySet;
    beforePhase3ForTest?: () => void;
}

export interface MaterializeDecision {
    value: boolean;
    reason: string | null;
}

export interface MaterializeM0Result {
    m0Bytes: Buffer;
    m0Text: string;
    m1Bytes: Buffer;
    m1Text: string;
    snapshotMarkers: M0SnapshotMarkers;
    renderedMemoryIds: number[];
}

export interface InjectM0M1Result {
    injected: boolean;
    m0RematerializedThisPass: boolean;
    materializationContentionRetryExhausted: boolean;
    decision: MaterializeDecision;
    m0Bytes: Buffer | null;
    m1Text: string | null;
}

export class MaterializeContentionError extends Error {
    readonly retries: number;
    readonly reason: string;

    constructor(args: { retries?: number; reason?: string } = {}) {
        super(args.reason ?? "m[0] materialization contention");
        this.name = "MaterializeContentionError";
        this.retries = args.retries ?? 0;
        this.reason = args.reason ?? "contention";
    }
}

export class RenderM1InvalidMarkersError extends Error {
    constructor(sessionId: string) {
        super(`Cannot render m[1] for ${sessionId}: missing cached m[0] snapshot markers`);
        this.name = "RenderM1InvalidMarkersError";
    }
}

// Compartment already carries p1..p4, importance, episodeType, legacy (v2 model B).
// Alias retained for readability at render call sites.
type M0Compartment = Compartment;

/**
 * The boundary (OpenCode message id) covered by a compartment set rendered into
 * m[0]+m[1] — the highest-sequence compartment's end message id, or null when
 * there are none / the latest has no stored boundary (legacy rows). The input
 * is ordered `sequence ASC`, so the last element is the latest compartment.
 */
function lastCompartmentBoundaryId(compartments: readonly M0Compartment[]): string | null {
    const last = compartments.at(-1);
    return last?.endMessageId && last.endMessageId.length > 0 ? last.endMessageId : null;
}

const DEFAULT_HISTORY_BUDGET_TOKENS = 60_000;
export const DEFAULT_MEMORY_BUDGET_TOKENS = 8_000;

/**
 * Token cost of the `<project-memory>` … `</project-memory>` wrapper itself,
 * seeded into the v2 trim accounting so the budget covers the whole injected
 * block (wrapper + lines), not just the line bodies.
 */
const MEMORY_BLOCK_WRAPPER_TOKENS = 6;
export const DEFAULT_USER_PROFILE_BUDGET_TOKENS = 4_000;
const M0_EMPTY_BODY = "<session-history></session-history>";
const M1_EMPTY_PLACEHOLDER =
    "<session-history-since>(no new content since last materialization)</session-history-since>";

export interface WorkspaceRenderContext {
    identities: string[];
    expandedIdentities: string[];
    ownIdentities: string[];
    shareCategories: string[] | null;
    namesByIdentity: Map<string, string>;
    canonicalIdentityByStoredPath: Map<string, string>;
    isWorkspaced: boolean;
}

export interface MemoryRenderOptions {
    sourceNameByMemoryId?: ReadonlyMap<number, string>;
}

function resolveWorkspaceRenderContext(args: {
    db: Database;
    projectPath?: string;
    workspaceIdentitySet?: WorkspaceIdentitySet;
}): WorkspaceRenderContext {
    if (!args.projectPath) {
        return {
            identities: [],
            expandedIdentities: [],
            ownIdentities: [],
            shareCategories: null,
            namesByIdentity: new Map(),
            canonicalIdentityByStoredPath: new Map(),
            isWorkspaced: false,
        };
    }
    const identitySet =
        args.workspaceIdentitySet ?? resolveWorkspaceIdentitySet(args.db, args.projectPath);
    const isWorkspaced = identitySet.identities.length > 1;
    const expanded = expandWorkspaceIdentitySetWithAliases(args.db, identitySet.identities);
    const expandedIdentities = isWorkspaced ? expanded.expandedIdentities : identitySet.identities;
    const canonicalIdentityByStoredPath = isWorkspaced
        ? expanded.canonicalIdentityByStoredPath
        : new Map(identitySet.identities.map((identity) => [identity, identity]));
    let ownIdentities = expandedIdentities.filter(
        (identity) => canonicalIdentityByStoredPath.get(identity) === args.projectPath,
    );
    if (ownIdentities.length === 0 && expandedIdentities.includes(args.projectPath)) {
        ownIdentities = [args.projectPath];
    }
    return {
        identities: identitySet.identities,
        expandedIdentities,
        ownIdentities,
        shareCategories: isWorkspaced
            ? resolveWorkspaceShareCategories(args.db, args.projectPath)
            : null,
        namesByIdentity: identitySet.namesByIdentity,
        canonicalIdentityByStoredPath,
        isWorkspaced,
    };
}

function sourceNamesForMemories(args: {
    memories: readonly Memory[];
    projectPath?: string;
    workspace: WorkspaceRenderContext;
}): Map<number, string> | undefined {
    if (!args.projectPath || !args.workspace.isWorkspaced) return undefined;
    const names = new Map<number, string>();
    for (const memory of args.memories) {
        const source = sourceNameForMemory(
            memory.projectPath,
            args.projectPath,
            args.workspace.identities,
            args.workspace.namesByIdentity,
            args.workspace.canonicalIdentityByStoredPath,
        );
        if (source) names.set(memory.id, source);
    }
    return names.size > 0 ? names : undefined;
}

function memoryCanonicalIdentity(memory: Memory, workspace: WorkspaceRenderContext): string | null {
    return resolveStoredPathWorkspaceIdentity(
        memory.projectPath,
        workspace.identities,
        workspace.canonicalIdentityByStoredPath,
    );
}

function memorySelectionOrder(left: Memory, right: Memory): number {
    if (left.status === "permanent" && right.status !== "permanent") return -1;
    if (right.status === "permanent" && left.status !== "permanent") return 1;
    const leftImportance = left.importance ?? Number.NEGATIVE_INFINITY;
    const rightImportance = right.importance ?? Number.NEGATIVE_INFINITY;
    const importanceDiff = rightImportance - leftImportance;
    if (importanceDiff !== 0) return importanceDiff;
    return left.id - right.id;
}

function memoryRenderOrder(left: Memory, right: Memory): number {
    const aPriority =
        (MEMORY_CATEGORY_ORDER_PRIORITY as Record<string, number>)[left.category] ??
        MEMORY_CATEGORY_ORDER_UNKNOWN;
    const bPriority =
        (MEMORY_CATEGORY_ORDER_PRIORITY as Record<string, number>)[right.category] ??
        MEMORY_CATEGORY_ORDER_UNKNOWN;
    const categoryDiff = aPriority - bPriority;
    if (categoryDiff !== 0) return categoryDiff;
    return left.id - right.id;
}

const maxCompartmentSeqStatements = new WeakMap<Database, PreparedStatement>();
const maxMemoryIdStatements = new WeakMap<Database, PreparedStatement>();
const legacyCompartmentCountStatements = new WeakMap<Database, PreparedStatement>();
const m0CompartmentStatements = new WeakMap<Database, PreparedStatement>();
const newCompartmentStatements = new WeakMap<Database, PreparedStatement>();

function cachedStatement(
    cache: WeakMap<Database, PreparedStatement>,
    db: Database,
    sql: string,
): PreparedStatement {
    let stmt = cache.get(db);
    if (!stmt) {
        stmt = db.prepare(sql);
        cache.set(db, stmt);
    }
    return stmt;
}

function numberFromRow(row: unknown, key: string): number {
    if (!row || typeof row !== "object") return 0;
    const value = (row as Record<string, unknown>)[key];
    return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function getMaxCompartmentSeq(db: Database, sessionId: string): number {
    const row = cachedStatement(
        maxCompartmentSeqStatements,
        db,
        "SELECT COALESCE(MAX(sequence), -1) AS s FROM compartments WHERE session_id = ?",
    ).get(sessionId);
    // -1 for an empty session, the real max sequence (>= 0) otherwise. The -1
    // sentinel is < 0 so it is distinct from the first real compartment (seq 0):
    // renderM1's readNewCompartments filters `sequence > maxSeq`, so an empty m[0]
    // baseline (maxCompartmentSeq = -1) includes the first compartment (seq 0) in
    // m[1]. New compartments are an m[1] delta, never a mustMaterialize trigger.
    return numberFromRow(row, "s");
}

function getMaxMemoryId(db: Database, projectPath: string | undefined): number {
    if (!projectPath) return 0;
    const row = cachedStatement(
        maxMemoryIdStatements,
        db,
        "SELECT COALESCE(MAX(id), 0) AS max_id FROM memories WHERE project_path = ?",
    ).get(projectPath);
    return numberFromRow(row, "max_id");
}

// v2: session_facts is retired as a render source (facts = promoted memories).
// The m[0] snapshot keeps a sessionFactsVersion field for shape stability, but
// it is pinned to 0 so fact changes never drive m[0] re-materialization —
// rendered bytes no longer depend on session_facts. (Avoids wasted rebuilds.)
// session_facts is a retired table (facts are promoted memories now); this
// branch is kept inert-safe but never fires. Do NOT rewire facts through here.
// See docs/AUDIT-KNOWN-ISSUES.md A14 (vestigial table, drop gated on min TUI).
function getSessionFactsVersion(_db: Database, _sessionId: string): number {
    return 0;
}

function getUpgradeState(db: Database, sessionId: string): string | null {
    const row = cachedStatement(
        legacyCompartmentCountStatements,
        db,
        "SELECT COUNT(*) AS count FROM compartments WHERE session_id = ? AND legacy = 1",
    ).get(sessionId);
    return numberFromRow(row, "count") > 0 ? "legacy" : "ready";
}

function getProjectMemoryEpoch(db: Database, projectPath: string | undefined): number {
    if (!projectPath) return 0;
    return getProjectState(db, projectPath)?.projectMemoryEpoch ?? 0;
}

function getGlobalUserProfileVersion(db: Database): number {
    return getProjectState(db, GLOBAL_USER_PROFILE_PROJECT_PATH)?.projectUserProfileVersion ?? 0;
}

export function readCurrentM0SnapshotMarkers(args: {
    db: Database;
    sessionId: string;
    projectPath?: string;
    projectDirectory?: string;
    hardSignals?: M0HardSignals;
    workspaceIdentitySet?: WorkspaceIdentitySet;
}): M0SnapshotMarkers {
    const projectDirectory = args.projectDirectory ?? args.projectPath ?? "";
    const hard = args.hardSignals ?? EMPTY_HARD_SIGNALS;
    const workspace = resolveWorkspaceRenderContext({
        db: args.db,
        projectPath: args.projectPath,
        workspaceIdentitySet: args.workspaceIdentitySet,
    });
    return {
        projectMemoryEpoch: getProjectMemoryEpoch(args.db, args.projectPath),
        workspaceFingerprint: workspace.isWorkspaced
            ? computeWorkspaceEpochFingerprint(args.db, workspace.identities)
            : null,
        projectUserProfileVersion: getGlobalUserProfileVersion(args.db),
        maxCompartmentSeq: getMaxCompartmentSeq(args.db, args.sessionId),
        maxMemoryId: workspace.isWorkspaced
            ? getMaxMemoryIdForProjects(
                  args.db,
                  workspace.expandedIdentities,
                  workspace.ownIdentities,
                  workspace.shareCategories,
              )
            : getMaxMemoryId(args.db, args.projectPath),
        maxMutationId: getMaxM0MutationId(args.db, args.sessionId) ?? 0,
        maxMemoryMutationId: workspace.isWorkspaced
            ? (getMaxMemoryMutationIdForProjects(args.db, workspace.expandedIdentities) ?? 0)
            : args.projectPath
              ? (getMaxMemoryMutationId(args.db, args.projectPath) ?? 0)
              : 0,
        projectDocsHash: projectDirectory ? computeProjectDocsHash(projectDirectory) : "",
        materializedAt: Date.now(),
        sessionFactsVersion: getSessionFactsVersion(args.db, args.sessionId),
        upgradeState: getUpgradeState(args.db, args.sessionId),
        systemHash: hard.systemHash,
        modelKey: hard.modelKey,
    };
}

function snapshotMarkersFromCachedM0(state: M0M1State): M0SnapshotMarkers | null {
    if (!state.cachedM0Bytes) return null;
    if (state.cachedM0ProjectMemoryEpoch === null) return null;
    if (state.cachedM0ProjectUserProfileVersion === null) return null;
    if (state.cachedM0MaxCompartmentSeq === null) return null;
    if (state.cachedM0MaxMemoryId === null) return null;
    if (state.cachedM0MaxMutationId === null) return null;
    if (state.cachedM0MaxMemoryMutationId === null) return null;
    if (state.cachedM0SessionFactsVersion === null) return null;
    return {
        projectMemoryEpoch: state.cachedM0ProjectMemoryEpoch,
        workspaceFingerprint: state.cachedM0WorkspaceFingerprint,
        projectUserProfileVersion: state.cachedM0ProjectUserProfileVersion,
        maxCompartmentSeq: state.cachedM0MaxCompartmentSeq,
        maxMemoryId: state.cachedM0MaxMemoryId,
        maxMutationId: state.cachedM0MaxMutationId,
        maxMemoryMutationId: state.cachedM0MaxMemoryMutationId,
        projectDocsHash: state.cachedM0ProjectDocsHash ?? "",
        materializedAt: state.cachedM0MaterializedAt ?? 0,
        sessionFactsVersion: state.cachedM0SessionFactsVersion,
        upgradeState: state.cachedM0UpgradeState,
        systemHash: state.cachedM0SystemHash ?? "",
        modelKey: state.cachedM0ModelKey ?? "",
    };
}

/**
 * The materialization decision, organized around the bust taxonomy:
 *
 *   SOFT+  — defer pass, nothing new: replay m[0] AND m[1] byte-identical.
 *   SOFT   — exec / deferred-consume pass: m[1] re-renders (new compartments,
 *            new memories, new user-profile ride the m[1] delta), m[0] stays.
 *   HARD   — the provider cache is already dead (idle>TTL, model/system/tools
 *            changed) OR a genuine m[0] *content* marker changed: fold m[1] into
 *            m[0], re-run decay, reset m[1].
 *
 * `mustMaterialize` returns true ONLY for HARD. New compartments and additive
 * user-profile/memory changes are deliberately NOT triggers — they are m[1]
 * deltas (see renderM1) and must never mutate the m[0] baseline. That is the
 * whole point of the m[0]=frozen-prefix / m[1]=volatile-delta split: a routine
 * historian publish must keep the Anthropic prompt-cache prefix intact.
 */
export function mustMaterialize(args: {
    db: Database;
    sessionId: string;
    state: M0M1State;
    projectPath?: string;
    projectDirectory?: string;
    hardSignals?: M0HardSignals;
    workspaceIdentitySet?: WorkspaceIdentitySet;
}): MaterializeDecision {
    if (!args.state.cachedM0Bytes) return { value: true, reason: "first_render" };
    if (!args.state.cachedM1Bytes) return { value: true, reason: "cached_m1_missing" };
    const hard = args.hardSignals ?? EMPTY_HARD_SIGNALS;
    // `current.workspaceFingerprint` is resolved inside readCurrentM0SnapshotMarkers
    // (it resolves its own workspace context); the HARD memory gate below keys on
    // that vs the cached fingerprint, so no local workspace context is needed here.
    const current = readCurrentM0SnapshotMarkers(args);

    // ── HARD: provider-side cache eviction (the cache was already dead) ──
    // Folding m[1] into m[0] here is "free" — the prefix is being re-cached
    // regardless. A non-empty current signal that differs from the captured
    // baseline marker means a real change; an empty current signal means
    // "unknown this pass" and is never treated as a change (avoids spurious
    // folds before the signal is known).
    if (hard.modelKey !== "" && hard.modelKey !== (args.state.cachedM0ModelKey ?? "")) {
        return { value: true, reason: "model_change" };
    }
    if (hard.systemHash !== "" && hard.systemHash !== (args.state.cachedM0SystemHash ?? "")) {
        return { value: true, reason: "system_hash" };
    }
    // Idle > TTL: the provider evicted the cache while the user was away. Guard
    // for idempotence across a multi-pass "came back" turn: cacheExpired stays
    // true on every pass until lastResponseTime updates at end-of-response, so
    // fold only when the last completed response is newer than our last
    // materialization. After the fold, materializedAt = Date.now() exceeds the
    // pre-expiry lastResponseTime, so subsequent passes this turn skip; the next
    // idle-after-response re-arms naturally. Self-consuming, no extra column.
    if (
        hard.cacheExpired &&
        hard.lastResponseTime > 0 &&
        hard.lastResponseTime > (args.state.cachedM0MaterializedAt ?? 0)
    ) {
        return { value: true, reason: "ttl_idle" };
    }

    // ── HARD: genuine m[0] CONTENT change (the rendered baseline bytes differ) ──
    // Compare the workspace fingerprint whenever EITHER the cached baseline or
    // the current pass is workspaced — keying only on current `isWorkspaced`
    // would miss the workspace→single transition: a cached union m[0] whose
    // session just left its workspace would fall through to the integer-epoch
    // compare and keep rendering the stale union if a membership bump were
    // missed. Mirrors renderM1's soft-refresh gate and Pi's mustMaterializePi.
    if (
        current.workspaceFingerprint !== null ||
        (args.state.cachedM0WorkspaceFingerprint ?? null) !== null
    ) {
        if ((args.state.cachedM0WorkspaceFingerprint ?? null) !== current.workspaceFingerprint) {
            return { value: true, reason: "project_memory_epoch" };
        }
    } else if (args.state.cachedM0ProjectMemoryEpoch !== current.projectMemoryEpoch) {
        return { value: true, reason: "project_memory_epoch" };
    }
    // NOTE: project_user_profile_version is deliberately NOT a trigger. Additive
    // user-profile promotions surface in m[1] via renderM1's <new-user-profile>
    // delta (version-watermark), exactly like new compartments and memories. A
    // version change must not fold m[0]; the delta reconciles into m[0] on the
    // next HARD fold. Destructive profile edits route through the same delta plus
    // the project_memory_epoch path for external (dashboard) mutations.
    //
    // NOTE: max_compartment_seq is deliberately NOT a trigger. New compartments
    // are the canonical m[1] delta (renderM1 -> readNewCompartments WHERE
    // sequence > cachedM0Seq). Folding m[0] on every historian publish would bust
    // the prompt-cache prefix on a routine background publish — the exact bug the
    // m[0]/m[1] split exists to prevent. They fold into m[0] only on a HARD bust.
    //
    // NOTE: maxMemoryId is NOT a trigger. Additive memory writes surface in m[1]
    // via the maxMemoryId watermark; memory mutations use the m[1] reconcile
    // cursor. max_mutation_id (structural compartment delete/merge/recomp) IS a
    // trigger because it changes the rendered m[0] baseline content.
    //
    // NOTE: projectDocsHash is deliberately NOT a trigger. Project docs are part
    // of m[0], but docs-only edits must not evict the cached prefix; materializeM0
    // reads fresh docs whenever a natural HARD fold happens and stores that hash
    // with the bytes it actually rendered.
    if (args.state.cachedM0MaxMutationId !== current.maxMutationId) {
        return { value: true, reason: "max_mutation_id" };
    }
    if ((args.state.cachedM0UpgradeState ?? null) !== current.upgradeState) {
        return { value: true, reason: "upgrade_state" };
    }
    return { value: false, reason: null };
}

export interface TrimMemoriesResultV2 {
    selected: Memory[];
    renderOrder: Memory[];
}

export function trimMemoriesToBudgetV2(
    sessionId: string,
    memories: Memory[],
    budgetTokens: number,
    renderOptions: MemoryRenderOptions = {},
): TrimMemoriesResultV2 {
    const selectionOrder = [...memories].sort(memorySelectionOrder);

    const selected: Memory[] = [];
    // Seed with the <project-memory> wrapper cost so the budget covers the whole
    // injected block, not just the line bodies.
    let usedTokens = MEMORY_BLOCK_WRAPPER_TOKENS;
    for (const memory of selectionOrder) {
        // Measure the EXACT v2 line that renderMemoryBlockV2 emits (with
        // id/category/importance attributes). Using a lighter shape here
        // under-counts and lets the rendered block overshoot the budget
        // (e.g. a v1 "- content" estimate fit 202 memories at ~8K while the v2
        // render actually injected ~11.3K against a 10K budget).
        const memoryTokens = estimateTokens(
            renderMemoryLineV2(memory, renderOptions.sourceNameByMemoryId?.get(memory.id)),
        );
        if (usedTokens + memoryTokens > budgetTokens) continue;
        selected.push(memory);
        usedTokens += memoryTokens;
    }

    if (selected.length < memories.length) {
        sessionLog(
            sessionId,
            `v2 trimmed memories from ${memories.length} to ${selected.length} to fit injection budget of ${budgetTokens} tokens`,
        );
    }

    const renderOrder = [...selected].sort(memoryRenderOrder);

    return { selected, renderOrder };
}

export function trimWorkspaceMemoriesToBudgetV2(
    sessionId: string,
    memories: Memory[],
    budgetTokens: number,
    workspace: WorkspaceRenderContext,
    renderOptions: MemoryRenderOptions = {},
): TrimMemoriesResultV2 {
    if (!workspace.isWorkspaced) {
        return trimMemoriesToBudgetV2(sessionId, memories, budgetTokens, renderOptions);
    }

    const selected: Memory[] = [];
    const selectedIds = new Set<number>();
    let usedTokens = MEMORY_BLOCK_WRAPPER_TOKENS;
    const tokenCost = (memory: Memory) =>
        estimateTokens(
            renderMemoryLineV2(memory, renderOptions.sourceNameByMemoryId?.get(memory.id)),
        );
    const trySelect = (memory: Memory): boolean => {
        if (selectedIds.has(memory.id)) return false;
        const tokens = tokenCost(memory);
        if (usedTokens + tokens > budgetTokens) return false;
        selected.push(memory);
        selectedIds.add(memory.id);
        usedTokens += tokens;
        return true;
    };

    for (const memory of memories
        .filter((candidate) => candidate.status === "permanent")
        .sort(memorySelectionOrder)) {
        trySelect(memory);
    }

    const remainingAfterPermanent = Math.max(0, budgetTokens - usedTokens);
    const floorTokens = remainingAfterPermanent / Math.max(1, workspace.identities.length);
    const byIdentity = new Map<string, Memory[]>();
    for (const memory of memories) {
        if (memory.status === "permanent") continue;
        const identity = memoryCanonicalIdentity(memory, workspace);
        if (!identity) continue;
        const list = byIdentity.get(identity) ?? [];
        list.push(memory);
        byIdentity.set(identity, list);
    }

    for (const identity of workspace.identities) {
        let memberTokens = 0;
        const candidates = (byIdentity.get(identity) ?? []).sort(memorySelectionOrder);
        for (const memory of candidates) {
            if (selectedIds.has(memory.id)) continue;
            const tokens = tokenCost(memory);
            if (memberTokens + tokens > floorTokens) continue;
            if (usedTokens + tokens > budgetTokens) continue;
            selected.push(memory);
            selectedIds.add(memory.id);
            usedTokens += tokens;
            memberTokens += tokens;
        }
    }

    const remaining = memories
        .filter((memory) => !selectedIds.has(memory.id))
        .sort(memorySelectionOrder);
    for (const memory of remaining) {
        trySelect(memory);
    }

    if (selected.length < memories.length) {
        sessionLog(
            sessionId,
            `v2 trimmed memories from ${memories.length} to ${selected.length} to fit injection budget of ${budgetTokens} tokens`,
        );
    }

    return { selected, renderOrder: [...selected].sort(memoryRenderOrder) };
}

function safeGetActiveUserMemories(db: Database): UserMemory[] {
    try {
        return getActiveUserMemories(db);
    } catch (error) {
        if (String(error).includes("no such table: user_memories")) return [];
        throw error;
    }
}

export function trimUserMemoriesToBudget(
    memories: UserMemory[],
    budgetTokens: number,
): UserMemory[] {
    const selected: UserMemory[] = [];
    let usedTokens = 0;
    for (const memory of memories) {
        const tokens = estimateTokens(`- ${memory.content}`) + 4;
        if (usedTokens + tokens > budgetTokens) continue;
        selected.push(memory);
        usedTokens += tokens;
    }
    return selected;
}

function readM0Compartments(db: Database, sessionId: string): M0Compartment[] {
    const rows = cachedStatement(
        m0CompartmentStatements,
        db,
        `SELECT id, session_id, sequence, start_message, end_message, start_message_id,
                end_message_id, title, content, p1, p2, p3, p4, episode_type,
                created_at, importance, legacy
           FROM compartments
          WHERE session_id = ?
          ORDER BY sequence ASC`,
    ).all(sessionId) as Array<Record<string, unknown>>;

    return rows.map(rowToM0Compartment);
}

function nullableString(value: unknown): string | null {
    return typeof value === "string" ? value : null;
}

function rowToM0Compartment(row: Record<string, unknown>): M0Compartment {
    return {
        id: Number(row.id ?? 0),
        sessionId: String(row.session_id ?? ""),
        sequence: Number(row.sequence ?? 0),
        startMessage: Number(row.start_message ?? 0),
        endMessage: Number(row.end_message ?? 0),
        startMessageId: String(row.start_message_id ?? ""),
        endMessageId: String(row.end_message_id ?? ""),
        title: String(row.title ?? ""),
        content: String(row.content ?? ""),
        p1: nullableString(row.p1),
        p2: nullableString(row.p2),
        p3: nullableString(row.p3),
        p4: nullableString(row.p4),
        importance: Number(row.importance ?? 50),
        episodeType: nullableString(row.episode_type),
        legacy: Number(row.legacy ?? 0),
        createdAt: Number(row.created_at ?? 0),
    };
}

function readNewCompartments(
    db: Database,
    sessionId: string,
    afterSequence: number,
): M0Compartment[] {
    const rows = cachedStatement(
        newCompartmentStatements,
        db,
        `SELECT id, session_id, sequence, start_message, end_message, start_message_id,
                end_message_id, title, content, p1, p2, p3, p4, episode_type,
                created_at, importance, legacy
           FROM compartments
          WHERE session_id = ? AND sequence > ?
          ORDER BY sequence ASC`,
    ).all(sessionId, afterSequence) as Array<Record<string, unknown>>;
    return rows.map(rowToM0Compartment);
}

function readNewMemoriesForM1(
    db: Database,
    projectPath: string | undefined,
    afterId: number,
    // Expiry cutoff is FROZEN to the m[0] materialization timestamp, NOT live
    // Date.now(). Defer passes replay the same markers (same materializedAt), so
    // the set of "not-yet-expired" memories rendered into m[1] stays byte-stable
    // until the next materialization. Using live Date.now() here would let a
    // memory crossing its expires_at between two defer passes silently change
    // m[1] bytes with no DB mutation — a cache-stability (Anthropic prefix) bust.
    expiryCutoff: number,
): Memory[] {
    if (!projectPath) return [];
    const rows = db
        .prepare(
            `SELECT ${getMemorySelectColumns(db)}
               FROM memories
              WHERE project_path = ?
                AND id > ?
                AND status IN ('active', 'permanent')
                AND (expires_at IS NULL OR expires_at > ?)
              ORDER BY ${MEMORY_CATEGORY_ORDER_SQL}, id ASC`,
        )
        .all(projectPath, afterId, expiryCutoff)
        .filter(isMemoryRow);
    return rows.map((row) => ({ ...row }));
}

/**
 * Render ONE memory's v2 line exactly as it lands in the <project-memory> block.
 * Shared by renderMemoryBlockV2 (the wire render) and trimMemoriesToBudgetV2
 * (the budget accounting) so the budget is measured against the SAME bytes that
 * get injected — including the id/category/importance attributes. Measuring a
 * lighter shape (e.g. "- content") under-counts and lets the injected block
 * exceed the configured budget.
 */
export function renderMemoryLineV2(memory: Memory, sourceName?: string): string {
    const sourceAttr = sourceName ? ` source="${escapeXmlAttr(sourceName)}"` : "";
    return `  <memory id="${memory.id}" category="${escapeXmlAttr(memory.category)}"${sourceAttr} importance="${memory.importance ?? 50}">${escapeXmlContent(memory.content)}</memory>`;
}

export function renderMemoryBlockV2(
    memories: Memory[],
    wrapper = "project-memory",
    renderOptions: MemoryRenderOptions = {},
): string {
    if (memories.length === 0) return "";
    const lines = [`<${wrapper}>`];
    for (const memory of memories) {
        lines.push(renderMemoryLineV2(memory, renderOptions.sourceNameByMemoryId?.get(memory.id)));
    }
    lines.push(`</${wrapper}>`);
    return lines.join("\n");
}

function renderUserProfileBlock(memories: UserMemory[], wrapper = "user-profile"): string {
    if (memories.length === 0) return "";
    const lines = [`<${wrapper}>`];
    for (const memory of memories) {
        lines.push(`- ${escapeXmlContent(memory.content)}`);
    }
    lines.push(`</${wrapper}>`);
    return lines.join("\n");
}

/**
 * v2 decayed session-history rendering delegates entirely to the shared
 * `decay-render` module (which uses the validated `decay-curve` formula). This
 * keeps OpenCode and Pi byte-identical and ensures the council-validated decay
 * math is the single source of truth — no local approximation lives here.
 *
 * Facts are NOT a render input (v2 faithful: facts = promoted memories).
 */
function renderSessionHistoryWithDecay(args: {
    compartments: M0Compartment[];
    historyBudgetTokens: number;
}): string {
    return renderDecayedCompartments({
        compartments: args.compartments,
        historyBudgetTokens: args.historyBudgetTokens,
    });
}

export function renderM0(args: {
    projectDocs: string;
    userProfileBaseline: UserMemory[];
    compartments: M0Compartment[];
    memories: Memory[];
    facts: SessionFact[];
    memoryRenderOptions?: MemoryRenderOptions;
    historyBudgetTokens?: number;
    userProfileBudgetTokens?: number;
    decayPressureMultiplier?: number;
}): string {
    const sections: string[] = [];
    if (args.projectDocs.length > 0) sections.push(args.projectDocs);
    const userProfile = renderUserProfileBlock(
        trimUserMemoriesToBudget(
            args.userProfileBaseline,
            args.userProfileBudgetTokens ?? DEFAULT_USER_PROFILE_BUDGET_TOKENS,
        ),
    );
    if (userProfile) sections.push(userProfile);

    // The +15% drift "pressure multiplier" maps to a proportionally tighter
    // effective budget (lower budget → higher curve pressure → more demotion),
    // keeping decay-curve.ts the single source of pressure math.
    const baseBudget = args.historyBudgetTokens ?? DEFAULT_HISTORY_BUDGET_TOKENS;
    const effectiveBudget = baseBudget / Math.max(1, args.decayPressureMultiplier ?? 1);
    const sessionHistory = renderSessionHistoryWithDecay({
        compartments: args.compartments,
        historyBudgetTokens: effectiveBudget,
    });
    sections.push(
        sessionHistory.length > 0
            ? `<session-history>\n${sessionHistory}\n</session-history>`
            : M0_EMPTY_BODY,
    );

    const memoriesBlock = renderMemoryBlockV2(
        args.memories,
        "project-memory",
        args.memoryRenderOptions,
    );
    if (memoriesBlock) sections.push(memoriesBlock);
    return sections.join("\n\n").trim();
}

function applyMarkersToState(
    state: M0M1State,
    m0Bytes: Buffer,
    markers: M0SnapshotMarkers,
    m1Bytes?: Buffer,
): void {
    state.cachedM0Bytes = m0Bytes;
    if (m1Bytes) state.cachedM1Bytes = m1Bytes;
    state.cachedM0ProjectMemoryEpoch = markers.projectMemoryEpoch;
    state.cachedM0WorkspaceFingerprint = markers.workspaceFingerprint;
    state.cachedM0ProjectUserProfileVersion = markers.projectUserProfileVersion;
    state.cachedM0MaxCompartmentSeq = markers.maxCompartmentSeq;
    state.cachedM0MaxMemoryId = markers.maxMemoryId;
    state.cachedM0MaxMutationId = markers.maxMutationId;
    state.cachedM0MaxMemoryMutationId = markers.maxMemoryMutationId;
    state.cachedM0ProjectDocsHash = markers.projectDocsHash;
    state.cachedM0MaterializedAt = markers.materializedAt;
    state.cachedM0SessionFactsVersion = markers.sessionFactsVersion;
    state.cachedM0UpgradeState = markers.upgradeState;
    // HARD-bust markers must be mirrored into the flat state fields too: the next
    // pass's mustMaterialize reads state.cachedM0SystemHash/ModelKey
    // directly (not snapshotMarkers). Omitting them here leaves the flat fields at
    // their pre-materialize values until a DB reload re-syncs them, which would
    // re-fire the same HARD trigger on the very next pass (double-fold).
    state.cachedM0SystemHash = markers.systemHash;
    state.cachedM0ModelKey = markers.modelKey;
    state.snapshotMarkers = markers;
}

/**
 * Real-tokenizer size of ONLY the <session-history> slice of a rendered m[0].
 *
 * The over-budget tightening loop must compare the history block against the
 * history budget — NOT the whole m[0]. m[0] also carries <project-docs>,
 * <user-profile>, and <project-memory>, each with its own budget; charging
 * those fixed blocks against the history budget falsely inflates measured cost,
 * over-tightens decay pressure, and starves session-history (e.g. project-docs
 * ~20K eating into a 98K history budget collapsed the effective budget to ~73K,
 * archiving ~157 extra compartments). Returns 0 when no session-history slice is
 * present (empty-history placeholder), so the loop never fires on empty history.
 */
function historySliceTokens(m0Text: string): number {
    const slice = extractM0Block(m0Text, "session-history");
    return slice ? estimateTokens(slice) : 0;
}

export function materializeM0(options: M0M1RenderOptions): MaterializeM0Result {
    const projectPath = options.projectPath;
    const projectDirectory = options.projectDirectory ?? projectPath ?? "";
    let snapshotMarkers: M0SnapshotMarkers;
    let compartments: M0Compartment[] = [];
    let facts: SessionFact[] = [];
    let memories: Memory[] = [];
    let userMemories: UserMemory[] = [];
    let workspace = resolveWorkspaceRenderContext({
        db: options.db,
        projectPath,
        workspaceIdentitySet: options.workspaceIdentitySet,
    });
    let docs: { renderedBlock: string; canonicalHash: string } = {
        renderedBlock: "",
        canonicalHash: "",
    };

    options.db.exec("BEGIN");
    try {
        workspace = resolveWorkspaceRenderContext({
            db: options.db,
            projectPath,
            workspaceIdentitySet: options.workspaceIdentitySet,
        });
        snapshotMarkers = readCurrentM0SnapshotMarkers({
            db: options.db,
            sessionId: options.sessionId,
            projectPath,
            projectDirectory,
            hardSignals: options.hardSignals,
            workspaceIdentitySet: {
                identities: workspace.identities,
                namesByIdentity: workspace.namesByIdentity,
            },
        });
        docs = projectDirectory
            ? readProjectDocsCanonical(projectDirectory)
            : { renderedBlock: "", canonicalHash: "" };
        snapshotMarkers.projectDocsHash = docs.canonicalHash;
        compartments = readM0Compartments(options.db, options.sessionId);
        // v2 faithful facts: session_facts is retired as a render source (facts
        // promote to project memory, rendered below via `memories`). Keep `facts`
        // empty so renderSessionHistoryWithDecay never emits a <session_facts>
        // block and no stale pre-v2 rows leak into m[0].
        facts = [];
        memories = projectPath
            ? workspace.isWorkspaced
                ? getMemoriesByProjects(
                      options.db,
                      workspace.expandedIdentities,
                      ["active", "permanent"],
                      Date.now(),
                      workspace.ownIdentities,
                      workspace.shareCategories,
                  )
                : getMemoriesByProject(options.db, projectPath, ["active", "permanent"])
            : [];
        userMemories = safeGetActiveUserMemories(options.db);
        options.db.exec("COMMIT");
    } catch (error) {
        try {
            options.db.exec("ROLLBACK");
        } catch {
            // ignore rollback failures from an already-closed transaction
        }
        throw error;
    }

    const memoryBudget = options.memoryInjectionBudgetTokens ?? DEFAULT_MEMORY_BUDGET_TOKENS;
    const memoryRenderOptions: MemoryRenderOptions = {
        sourceNameByMemoryId: sourceNamesForMemories({
            memories,
            projectPath,
            workspace,
        }),
    };
    const trimmed = workspace.isWorkspaced
        ? trimWorkspaceMemoriesToBudgetV2(
              options.sessionId,
              memories,
              memoryBudget,
              workspace,
              memoryRenderOptions,
          )
        : trimMemoriesToBudgetV2(options.sessionId, memories, memoryBudget);
    let decayPressureMultiplier = 1;
    let m0Text = renderM0({
        projectDocs: docs.renderedBlock,
        userProfileBaseline: userMemories,
        compartments,
        memories: trimmed.renderOrder,
        facts,
        memoryRenderOptions,
        historyBudgetTokens: options.historyBudgetTokens ?? DEFAULT_HISTORY_BUDGET_TOKENS,
        userProfileBudgetTokens: options.userProfileBudgetTokens,
        decayPressureMultiplier,
    });

    let attempts = 0;
    const budget = options.historyBudgetTokens ?? DEFAULT_HISTORY_BUDGET_TOKENS;
    while (budget > 0 && historySliceTokens(m0Text) > budget * 1.05 && attempts < 3) {
        decayPressureMultiplier *= 1.15;
        m0Text = renderM0({
            projectDocs: docs.renderedBlock,
            userProfileBaseline: userMemories,
            compartments,
            memories: trimmed.renderOrder,
            facts,
            memoryRenderOptions,
            historyBudgetTokens: budget,
            userProfileBudgetTokens: options.userProfileBudgetTokens,
            decayPressureMultiplier,
        });
        attempts += 1;
    }

    if (m0Text.length === 0) m0Text = M0_EMPTY_BODY;
    const m0Bytes = Buffer.from(m0Text, "utf8");
    snapshotMarkers.materializedAt = Date.now();
    const renderedMemoryIds = trimmed.renderOrder.map((m) => m.id);
    const preRenderedKeyFilesBlock = preRenderKeyFilesBlock(options);
    const phase3ProjectDocsHash = projectDirectory ? computeProjectDocsHash(projectDirectory) : "";

    options.beforePhase3ForTest?.();

    let m1Text = M1_EMPTY_PLACEHOLDER;
    let m1Bytes = Buffer.from(m1Text, "utf8");
    options.db.exec("BEGIN IMMEDIATE");
    try {
        const currentWorkspace = resolveWorkspaceRenderContext({
            db: options.db,
            projectPath,
            workspaceIdentitySet: options.workspaceIdentitySet,
        });
        const current: M0SnapshotMarkers = {
            projectMemoryEpoch: getProjectMemoryEpoch(options.db, projectPath),
            workspaceFingerprint: currentWorkspace.isWorkspaced
                ? computeWorkspaceEpochFingerprint(options.db, currentWorkspace.identities)
                : null,
            projectUserProfileVersion: getGlobalUserProfileVersion(options.db),
            maxCompartmentSeq: getMaxCompartmentSeq(options.db, options.sessionId),
            maxMemoryId: currentWorkspace.isWorkspaced
                ? getMaxMemoryIdForProjects(
                      options.db,
                      currentWorkspace.expandedIdentities,
                      currentWorkspace.ownIdentities,
                      currentWorkspace.shareCategories,
                  )
                : getMaxMemoryId(options.db, projectPath),
            maxMutationId: getMaxM0MutationId(options.db, options.sessionId) ?? 0,
            maxMemoryMutationId: currentWorkspace.isWorkspaced
                ? (getMaxMemoryMutationIdForProjects(
                      options.db,
                      currentWorkspace.expandedIdentities,
                  ) ?? 0)
                : projectPath
                  ? (getMaxMemoryMutationId(options.db, projectPath) ?? 0)
                  : 0,
            projectDocsHash: phase3ProjectDocsHash,
            materializedAt: Date.now(),
            sessionFactsVersion: getSessionFactsVersion(options.db, options.sessionId),
            upgradeState: getUpgradeState(options.db, options.sessionId),
            // HARD-bust markers are flight-constant (system/tool/model identity of
            // THIS request) — they cannot change mid-materialization-transaction,
            // so carry the captured values and exclude them from the stale check.
            systemHash: snapshotMarkers.systemHash,
            modelKey: snapshotMarkers.modelKey,
        };
        // NOTE: maxMemoryId is deliberately EXCLUDED from this stale-check.
        // Additive memory writes (write/promote) do not invalidate the rendered
        // m[0]; they surface in m[1] via the maxMemoryId watermark. The memory
        // mutation cursor IS included here because a materialization pass must
        // reconcile every non-additive memory change up to its persisted cursor.
        const memoryEpochStale =
            current.workspaceFingerprint !== null || snapshotMarkers.workspaceFingerprint !== null
                ? current.workspaceFingerprint !== snapshotMarkers.workspaceFingerprint
                : current.projectMemoryEpoch !== snapshotMarkers.projectMemoryEpoch;
        const stale =
            memoryEpochStale ||
            current.projectUserProfileVersion !== snapshotMarkers.projectUserProfileVersion ||
            current.maxCompartmentSeq !== snapshotMarkers.maxCompartmentSeq ||
            current.maxMutationId !== snapshotMarkers.maxMutationId ||
            current.maxMemoryMutationId !== snapshotMarkers.maxMemoryMutationId ||
            current.sessionFactsVersion !== snapshotMarkers.sessionFactsVersion ||
            current.upgradeState !== snapshotMarkers.upgradeState;
        if (stale) {
            options.db.exec("ROLLBACK");
            throw new MaterializeContentionError({ reason: "snapshot changed before Phase 3" });
        }

        const m1Render = renderM1WithMetadata(
            {
                ...options,
                preRenderedKeyFilesBlock,
                workspaceIdentitySet: {
                    identities: workspace.identities,
                    namesByIdentity: workspace.namesByIdentity,
                },
            },
            snapshotMarkers,
            renderedMemoryIds,
        );
        m1Text = m1Render.text;
        m1Bytes = Buffer.from(m1Text, "utf8");

        persistCachedM0(options.db, options.sessionId, {
            m0Bytes,
            projectMemoryEpoch: snapshotMarkers.projectMemoryEpoch,
            workspaceFingerprint: snapshotMarkers.workspaceFingerprint,
            projectUserProfileVersion: snapshotMarkers.projectUserProfileVersion,
            maxCompartmentSeq: snapshotMarkers.maxCompartmentSeq,
            maxMemoryId: snapshotMarkers.maxMemoryId,
            maxMutationId: snapshotMarkers.maxMutationId,
            maxMemoryMutationId: snapshotMarkers.maxMemoryMutationId,
            m1Bytes,
            projectDocsHash: snapshotMarkers.projectDocsHash,
            materializedAt: snapshotMarkers.materializedAt,
            sessionFactsVersion: snapshotMarkers.sessionFactsVersion,
            upgradeState: snapshotMarkers.upgradeState,
            systemHash: snapshotMarkers.systemHash,
            modelKey: snapshotMarkers.modelKey,
        });

        // v2 path persists the rendered-memory identity itself. `memory_block_ids`
        // / `memory_block_count` are otherwise written ONLY by the dead legacy v1
        // render path, so without this they stay frozen at whatever the last legacy
        // render wrote — wrong sidebar "Injected" count AND a stale ctx_search
        // hide-already-visible filter after any memory change (e.g. the migration
        // delete+reinserts memories with NEW ids; the old ids linger here).
        // dogfood 2026-05-30: AFT showed "Injected 256" against 124 live memories,
        // all 256 ids deleted. Same transaction as the m[0] snapshot so the cached
        // bytes and their id manifest never diverge.
        options.db
            .prepare(
                "UPDATE session_meta SET memory_block_count = ?, memory_block_ids = ? WHERE session_id = ?",
            )
            .run(renderedMemoryIds.length, JSON.stringify(renderedMemoryIds), options.sessionId);

        // Persist the boundary the freshly-rendered m[0]+m[1] cover (the latest
        // compartment's end message id). A cold post-restart pass reads this to
        // trim the live tail to what the cached summary covers — never past it —
        // so a compartment published after this materialize keeps its raw
        // messages in the tail until an exec pass folds it into m[1]. Same
        // transaction as the m[0] snapshot so bytes and boundary never diverge.
        const baselineEndMessageId = lastCompartmentBoundaryId(compartments);
        options.db
            .prepare(
                "UPDATE session_meta SET cached_m0_last_baseline_end_message_id = ? WHERE session_id = ?",
            )
            .run(baselineEndMessageId, options.sessionId);

        options.db.exec("COMMIT");
    } catch (error) {
        try {
            options.db.exec("ROLLBACK");
        } catch {
            // already rolled back
        }
        throw error;
    }

    return { m0Bytes, m0Text, m1Bytes, m1Text, snapshotMarkers, renderedMemoryIds };
}

export function materializeWithRetry(
    options: M0M1RenderOptions,
    maxRetries = 3,
): MaterializeM0Result {
    let lastError: MaterializeContentionError | null = null;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return materializeM0(options);
        } catch (error) {
            if (!(error instanceof MaterializeContentionError)) throw error;
            lastError = error;
        }
    }
    throw new MaterializeContentionError({
        retries: maxRetries,
        reason: lastError?.reason ?? "m[0] materialization contention exhausted",
    });
}

function preRenderKeyFilesBlock(options: M0M1RenderOptions): string | null {
    if (!options.projectDirectory) return null;
    try {
        return buildKeyFilesBlock(options.db, options.projectDirectory, options.keyFiles) ?? null;
    } catch (error) {
        sessionLog(options.sessionId, "key-files render for m[1] failed:", error);
        return null;
    }
}

function renderedKeyFilesBlock(options: M0M1RenderOptions): string | null {
    if (options.preRenderedKeyFilesBlock !== undefined) {
        return options.preRenderedKeyFilesBlock;
    }
    return preRenderKeyFilesBlock(options);
}

function renderMemoryUpdatesBlock(args: {
    db: Database;
    projectPath?: string;
    workspace: WorkspaceRenderContext;
    afterId: number;
    renderedMemoryIds: readonly number[];
}): { block: string; count: number } {
    if (!args.projectPath || args.renderedMemoryIds.length === 0) {
        return { block: "", count: 0 };
    }

    const renderedIds = new Set(args.renderedMemoryIds);
    const mutations = args.workspace.isWorkspaced
        ? getMemoryMutationsForRenderByProjects(
              args.db,
              args.workspace.expandedIdentities,
              args.afterId,
              args.renderedMemoryIds,
          )
        : getMemoryMutationsForRender(
              args.db,
              args.projectPath,
              args.afterId,
              args.renderedMemoryIds,
          );
    if (mutations.length === 0) return { block: "", count: 0 };

    const lines = ["These memories changed since the snapshot below — trust these:"];
    for (const mutation of mutations) {
        if (mutation.mutationType === "update") {
            lines.push(
                `  <updated id="${mutation.targetMemoryId}">${escapeXmlContent(mutation.newContent ?? "")}</updated>`,
            );
            continue;
        }
        if (mutation.mutationType === "superseded") {
            if (mutation.supersededById !== null && renderedIds.has(mutation.supersededById)) {
                lines.push(
                    `  <superseded id="${mutation.targetMemoryId}" by="${mutation.supersededById}"/>`,
                );
            } else {
                lines.push(`  <removed id="${mutation.targetMemoryId}"/>`);
            }
            continue;
        }
        lines.push(`  <removed id="${mutation.targetMemoryId}"/>`);
    }

    return {
        block: `<memory-updates>\n${lines.join("\n")}\n</memory-updates>`,
        count: mutations.length,
    };
}

interface RenderM1Result {
    text: string;
    memoryUpdateCount: number;
}

function renderM1WithMetadata(
    options: M0M1RenderOptions,
    markers: M0SnapshotMarkers,
    renderedMemoryIds: readonly number[],
): RenderM1Result {
    if (!markers || markers.maxCompartmentSeq === undefined) {
        throw new RenderM1InvalidMarkersError(options.sessionId);
    }

    const blocks: string[] = [];
    const workspace = resolveWorkspaceRenderContext({
        db: options.db,
        projectPath: options.projectPath,
        workspaceIdentitySet: options.workspaceIdentitySet,
    });
    const keyFiles = renderedKeyFilesBlock(options);
    if (keyFiles) blocks.push(keyFiles);

    const memoryUpdates = renderMemoryUpdatesBlock({
        db: options.db,
        projectPath: options.projectPath,
        workspace,
        afterId: markers.maxMemoryMutationId,
        renderedMemoryIds,
    });
    if (memoryUpdates.block) blocks.push(memoryUpdates.block);

    const newCompartments = readNewCompartments(
        options.db,
        options.sessionId,
        markers.maxCompartmentSeq,
    );
    if (newCompartments.length > 0) {
        blocks.push(
            `<new-compartments>\n${newCompartments
                .map((compartment) => renderCompartmentAtTier(compartment, 1))
                .join("\n\n")}\n</new-compartments>`,
        );
    }

    const newMemories = workspace.isWorkspaced
        ? readNewMemoriesForM1Union(
              options.db,
              workspace.expandedIdentities,
              markers.maxMemoryId,
              // Freeze expiry to the materialization timestamp for defer-pass byte stability.
              markers.materializedAt,
              workspace.ownIdentities,
              workspace.shareCategories,
          )
        : readNewMemoriesForM1(
              options.db,
              options.projectPath,
              markers.maxMemoryId,
              // Freeze expiry to the materialization timestamp for defer-pass byte stability.
              markers.materializedAt,
          );
    const newMemoryRenderOptions: MemoryRenderOptions = {
        sourceNameByMemoryId: sourceNamesForMemories({
            memories: newMemories,
            projectPath: options.projectPath,
            workspace,
        }),
    };
    const trimmedNewMemories = trimMemoriesToBudgetV2(
        options.sessionId,
        newMemories,
        Math.max(
            1,
            Math.floor(
                (options.memoryInjectionBudgetTokens ?? DEFAULT_MEMORY_BUDGET_TOKENS) * 0.25,
            ),
        ),
        newMemoryRenderOptions,
    ).renderOrder;
    const newMemoriesBlock = renderMemoryBlockV2(
        trimmedNewMemories,
        "new-memories",
        newMemoryRenderOptions,
    );
    if (newMemoriesBlock) blocks.push(newMemoriesBlock);

    const currentUserProfileVersion = getGlobalUserProfileVersion(options.db);
    if (currentUserProfileVersion !== markers.projectUserProfileVersion) {
        const profileBlock = renderUserProfileBlock(
            trimUserMemoriesToBudget(
                safeGetActiveUserMemories(options.db),
                Math.max(
                    1,
                    Math.floor(
                        (options.userProfileBudgetTokens ?? DEFAULT_USER_PROFILE_BUDGET_TOKENS) *
                            0.25,
                    ),
                ),
            ),
            "new-user-profile",
        );
        if (profileBlock) blocks.push(profileBlock);
    }

    // v2 faithful facts: session_facts is retired as a render source. Fresh
    // facts reach the agent as promoted memories via the new-memories block
    // above (maxMemoryId watermark), not via a <session_facts> delta here.

    if (blocks.length === 0) {
        return { text: M1_EMPTY_PLACEHOLDER, memoryUpdateCount: memoryUpdates.count };
    }
    return {
        text: `<session-history-since>\n${blocks.join("\n")}\n</session-history-since>`,
        memoryUpdateCount: memoryUpdates.count,
    };
}

export function renderM1(
    options: M0M1RenderOptions,
    markers: M0SnapshotMarkers,
    renderedMemoryIds: readonly number[] = [],
): string {
    return renderM1WithMetadata(options, markers, renderedMemoryIds).text;
}

function decodeM0Bytes(bytes: Buffer | Uint8Array | null): string | null {
    if (!bytes) return null;
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("utf8");
}

interface CachedM0M1Row {
    cached_m0_bytes: Buffer | Uint8Array | null;
    cached_m1_bytes: Buffer | Uint8Array | null;
    cached_m0_project_memory_epoch: number | null;
    cached_m0_workspace_fingerprint: string | null;
    cached_m0_project_user_profile_version: number | null;
    cached_m0_max_compartment_seq: number | null;
    cached_m0_max_memory_id: number | null;
    cached_m0_max_mutation_id: number | null;
    cached_m0_max_memory_mutation_id: number | null;
    cached_m0_project_docs_hash: string | null;
    cached_m0_materialized_at: number | null;
    cached_m0_session_facts_version: number | null;
    cached_m0_upgrade_state: string | null;
    cached_m0_system_hash: string | null;
    cached_m0_model_key: string | null;
    memory_block_ids: string | null;
}

function toBuffer(value: Buffer | Uint8Array): Buffer {
    return Buffer.isBuffer(value)
        ? value
        : Buffer.from(value.buffer, value.byteOffset, value.byteLength);
}

function bufferEqualsNullable(
    left: Buffer | Uint8Array | null,
    right: Buffer | Uint8Array | null,
): boolean {
    if (left === null || right === null) return left === right;
    return toBuffer(left).equals(toBuffer(right));
}

function parseMemoryBlockIds(raw: string | null): number[] {
    if (!raw) return [];
    try {
        const parsed = JSON.parse(raw) as unknown;
        if (!Array.isArray(parsed)) return [];
        return parsed.filter((value): value is number => typeof value === "number");
    } catch {
        return [];
    }
}

function readCachedM0M1Row(db: Database, sessionId: string): CachedM0M1Row | null {
    return db
        .prepare(
            `SELECT cached_m0_bytes, cached_m1_bytes,
                    cached_m0_project_memory_epoch,
                    cached_m0_workspace_fingerprint,
                    cached_m0_project_user_profile_version,
                    cached_m0_max_compartment_seq,
                    cached_m0_max_memory_id,
                    cached_m0_max_mutation_id,
                    cached_m0_max_memory_mutation_id,
                    cached_m0_project_docs_hash,
                    cached_m0_materialized_at,
                    cached_m0_session_facts_version,
                    cached_m0_upgrade_state,
                    cached_m0_system_hash,
                    cached_m0_model_key,
                    memory_block_ids
               FROM session_meta
              WHERE session_id = ?`,
        )
        .get(sessionId) as CachedM0M1Row | null;
}

function markersFromCachedRow(row: CachedM0M1Row): M0SnapshotMarkers | null {
    if (!row.cached_m0_bytes) return null;
    if (row.cached_m0_project_memory_epoch === null) return null;
    if (row.cached_m0_project_user_profile_version === null) return null;
    if (row.cached_m0_max_compartment_seq === null) return null;
    if (row.cached_m0_max_memory_id === null) return null;
    if (row.cached_m0_max_mutation_id === null) return null;
    if (row.cached_m0_max_memory_mutation_id === null) return null;
    if (row.cached_m0_session_facts_version === null) return null;
    return {
        projectMemoryEpoch: row.cached_m0_project_memory_epoch,
        workspaceFingerprint: row.cached_m0_workspace_fingerprint,
        projectUserProfileVersion: row.cached_m0_project_user_profile_version,
        maxCompartmentSeq: row.cached_m0_max_compartment_seq,
        maxMemoryId: row.cached_m0_max_memory_id,
        maxMutationId: row.cached_m0_max_mutation_id,
        maxMemoryMutationId: row.cached_m0_max_memory_mutation_id,
        projectDocsHash: row.cached_m0_project_docs_hash ?? "",
        materializedAt: row.cached_m0_materialized_at ?? 0,
        sessionFactsVersion: row.cached_m0_session_facts_version,
        upgradeState: row.cached_m0_upgrade_state,
        systemHash: row.cached_m0_system_hash ?? "",
        modelKey: row.cached_m0_model_key ?? "",
    };
}

function cachedRowMatchesState(row: CachedM0M1Row, state: M0M1State): boolean {
    return (
        bufferEqualsNullable(row.cached_m0_bytes, state.cachedM0Bytes) &&
        row.cached_m0_project_memory_epoch === state.cachedM0ProjectMemoryEpoch &&
        (row.cached_m0_workspace_fingerprint ?? null) ===
            (state.cachedM0WorkspaceFingerprint ?? null) &&
        row.cached_m0_project_user_profile_version === state.cachedM0ProjectUserProfileVersion &&
        row.cached_m0_max_compartment_seq === state.cachedM0MaxCompartmentSeq &&
        row.cached_m0_max_memory_id === state.cachedM0MaxMemoryId &&
        row.cached_m0_max_mutation_id === state.cachedM0MaxMutationId &&
        row.cached_m0_max_memory_mutation_id === state.cachedM0MaxMemoryMutationId &&
        // Project-docs hash is inert for CAS decisions: byte-different m[0] rows
        // fail the buffer compare above, while hash-only drift with identical bytes
        // must still refresh m[1] against the current cached prefix.
        row.cached_m0_materialized_at === state.cachedM0MaterializedAt &&
        row.cached_m0_session_facts_version === state.cachedM0SessionFactsVersion &&
        (row.cached_m0_upgrade_state ?? null) === (state.cachedM0UpgradeState ?? null) &&
        (row.cached_m0_system_hash ?? "") === (state.cachedM0SystemHash ?? "") &&
        (row.cached_m0_model_key ?? "") === (state.cachedM0ModelKey ?? "")
    );
}

function applyCachedRowToState(state: M0M1State, row: CachedM0M1Row): void {
    const markers = markersFromCachedRow(row);
    if (!row.cached_m0_bytes || !row.cached_m1_bytes || !markers) {
        throw new RenderM1InvalidMarkersError(state.sessionId);
    }
    state.cachedM0Bytes = toBuffer(row.cached_m0_bytes);
    state.cachedM1Bytes = toBuffer(row.cached_m1_bytes);
    state.cachedM0ProjectMemoryEpoch = markers.projectMemoryEpoch;
    state.cachedM0WorkspaceFingerprint = markers.workspaceFingerprint;
    state.cachedM0ProjectUserProfileVersion = markers.projectUserProfileVersion;
    state.cachedM0MaxCompartmentSeq = markers.maxCompartmentSeq;
    state.cachedM0MaxMemoryId = markers.maxMemoryId;
    state.cachedM0MaxMutationId = markers.maxMutationId;
    state.cachedM0MaxMemoryMutationId = markers.maxMemoryMutationId;
    state.cachedM0ProjectDocsHash = markers.projectDocsHash;
    state.cachedM0MaterializedAt = markers.materializedAt;
    state.cachedM0SessionFactsVersion = markers.sessionFactsVersion;
    state.cachedM0UpgradeState = markers.upgradeState;
    state.cachedM0SystemHash = markers.systemHash;
    state.cachedM0ModelKey = markers.modelKey;
    state.snapshotMarkers = markers;
}

function replayCachedM1(state: M0M1State): string {
    if (!state.cachedM1Bytes) {
        throw new RenderM1InvalidMarkersError(state.sessionId);
    }
    return decodeM0Bytes(state.cachedM1Bytes) ?? M1_EMPTY_PLACEHOLDER;
}

function softRefreshCachedM1(options: M0M1RenderOptions): RenderM1Result {
    const preRenderedKeyFilesBlock = preRenderKeyFilesBlock(options);
    options.db.exec("BEGIN IMMEDIATE");
    try {
        const row = readCachedM0M1Row(options.db, options.sessionId);
        if (!row || !cachedRowMatchesState(row, options.state)) {
            options.db.exec("ROLLBACK");
            // Post-ROLLBACK fallback read is intentionally NOT wrapped in a
            // transaction: readCachedM0M1Row is a SINGLE atomic SELECT, so
            // SQLite guarantees m0/m1/markers all come from the same committed
            // row — a torn cross-column read is impossible. If another sibling
            // commits between ROLLBACK and this read we simply adopt that newer
            // (still self-consistent) row, which is correct. Wrapping a single
            // SELECT in BEGIN/COMMIT would add write-lock contention on this hot
            // path (every cache-busting pass) for zero consistency gain.
            const sibling = readCachedM0M1Row(options.db, options.sessionId);
            if (!sibling) throw new RenderM1InvalidMarkersError(options.sessionId);
            applyCachedRowToState(options.state, sibling);
            return { text: replayCachedM1(options.state), memoryUpdateCount: 0 };
        }

        const markers = markersFromCachedRow(row);
        if (!markers) throw new RenderM1InvalidMarkersError(options.sessionId);
        const renderedMemoryIds = parseMemoryBlockIds(row.memory_block_ids);
        const rendered = renderM1WithMetadata(
            { ...options, preRenderedKeyFilesBlock },
            markers,
            renderedMemoryIds,
        );
        const m1Bytes = Buffer.from(rendered.text, "utf8");
        // Advance the persisted baseline boundary too: soft-refresh re-renders
        // m[1] to cover every compartment up to the latest, so the boundary the
        // cached summary covers moves forward with it. Keeping it in sync here is
        // what lets a later cold post-restart defer pass trim correctly.
        const baselineEndMessageId = getLastCompartmentEndMessageId(options.db, options.sessionId);
        options.db
            .prepare(
                "UPDATE session_meta SET cached_m1_bytes = ?, cached_m0_last_baseline_end_message_id = ? WHERE session_id = ?",
            )
            .run(m1Bytes, baselineEndMessageId, options.sessionId);
        options.db.exec("COMMIT");
        options.state.cachedM1Bytes = m1Bytes;
        options.state.snapshotMarkers = markers;
        return rendered;
    } catch (error) {
        try {
            options.db.exec("ROLLBACK");
        } catch {
            // already rolled back
        }
        throw error;
    }
}

function prependM0M1Messages(
    sessionId: string,
    messages: MessageLike[],
    m0Text: string,
    m1Text: string,
): void {
    // `synthetic: true` marks these as injected context, not real user turns.
    // OpenCode's `toModelMessagesEffect` filters on `ignored` (NOT `synthetic`),
    // so the blocks STILL reach the model — but its title-generation gate
    // (`ensureTitle`) counts a message as a real user turn only when not every
    // part is synthetic, and skips titling unless exactly one real user message
    // exists. Without this flag, m[0]+m[1] add two phantom user turns on the
    // first message and permanently suppress the session's auto-title (issue
    // #129). Must NOT use `ignored` here — that would strip the history
    // injection from the real model call.
    messages.unshift(
        {
            info: { role: "user", sessionID: sessionId },
            parts: [
                {
                    type: "text",
                    text: m0Text.length > 0 ? m0Text : M0_EMPTY_BODY,
                    synthetic: true,
                },
            ],
        },
        {
            info: { role: "user", sessionID: sessionId },
            parts: [{ type: "text", text: m1Text, synthetic: true }],
        },
    );
}

/**
 * Render a fresh m[0] from current DB state WITHOUT persisting it or taking the
 * materialize lock. Last-resort fallback for injectM0M1 when materialization
 * loses the lock (contention exhausted) AND there is no cached baseline to reuse
 * — e.g. the cache was cleared this pass by a history refresh and then a sibling
 * process held the lock. Dropping injection would send the model zero session
 * history; rendering fresh (un-cached) keeps history present for this pass while
 * the next pass re-materializes and persists. Mirrors Pi's renderM0Pi fallback.
 * Uses a plain read (no BEGIN IMMEDIATE) since we are explicitly NOT persisting.
 */
function renderFreshM0NonPersisted(options: M0M1RenderOptions): {
    m0Bytes: Buffer;
    snapshotMarkers: M0SnapshotMarkers;
    renderedMemoryIds: number[];
} {
    const projectPath = options.projectPath;
    const projectDirectory = options.projectDirectory;
    const workspace = resolveWorkspaceRenderContext({
        db: options.db,
        projectPath,
        workspaceIdentitySet: options.workspaceIdentitySet,
    });
    const snapshotMarkers = readCurrentM0SnapshotMarkers({
        db: options.db,
        sessionId: options.sessionId,
        projectPath,
        projectDirectory,
        workspaceIdentitySet: {
            identities: workspace.identities,
            namesByIdentity: workspace.namesByIdentity,
        },
    });
    const docs = projectDirectory
        ? readProjectDocsCanonical(projectDirectory)
        : { renderedBlock: "", canonicalHash: "" };
    snapshotMarkers.projectDocsHash = docs.canonicalHash;
    // CACHE STABILITY: materializedAt feeds the m[1] memory-expiry cutoff
    // (renderM1). It MUST be stable across consecutive fallback passes, or two
    // defer passes that straddle a memory's expires_at would render different
    // m[1] bytes with zero DB mutation. Never use live Date.now() here. Reuse
    // the last persisted materialization timestamp; if none exists (cache fully
    // cleared this pass), use 0 (stable: renders all memories with no expiry
    // filtering, deterministic across passes — matches Pi fallback).
    snapshotMarkers.materializedAt = options.state.cachedM0MaterializedAt ?? 0;
    const compartments = readM0Compartments(options.db, options.sessionId);
    // Use the SAME frozen cutoff for the baseline memory read as m[1] does, so a
    // memory crossing expires_at between two fallback passes can't shift the m[0]
    // baseline bytes either (live Date.now() default would reintroduce drift).
    const memories = projectPath
        ? workspace.isWorkspaced
            ? getMemoriesByProjects(
                  options.db,
                  workspace.expandedIdentities,
                  ["active", "permanent"],
                  snapshotMarkers.materializedAt,
                  workspace.ownIdentities,
                  workspace.shareCategories,
              )
            : getMemoriesByProject(
                  options.db,
                  projectPath,
                  ["active", "permanent"],
                  snapshotMarkers.materializedAt,
              )
        : [];
    const userMemories = safeGetActiveUserMemories(options.db);
    const memoryBudget = options.memoryInjectionBudgetTokens ?? DEFAULT_MEMORY_BUDGET_TOKENS;
    const memoryRenderOptions: MemoryRenderOptions = {
        sourceNameByMemoryId: sourceNamesForMemories({
            memories,
            projectPath,
            workspace,
        }),
    };
    const trimmed = workspace.isWorkspaced
        ? trimWorkspaceMemoriesToBudgetV2(
              options.sessionId,
              memories,
              memoryBudget,
              workspace,
              memoryRenderOptions,
          )
        : trimMemoriesToBudgetV2(options.sessionId, memories, memoryBudget);
    const budget = options.historyBudgetTokens ?? DEFAULT_HISTORY_BUDGET_TOKENS;
    let decayPressureMultiplier = 1;
    let m0Text = renderM0({
        projectDocs: docs.renderedBlock,
        userProfileBaseline: userMemories,
        compartments,
        memories: trimmed.renderOrder,
        facts: [],
        memoryRenderOptions,
        historyBudgetTokens: budget,
        userProfileBudgetTokens: options.userProfileBudgetTokens,
        decayPressureMultiplier,
    });
    let attempts = 0;
    while (budget > 0 && historySliceTokens(m0Text) > budget * 1.05 && attempts < 3) {
        decayPressureMultiplier *= 1.15;
        m0Text = renderM0({
            projectDocs: docs.renderedBlock,
            userProfileBaseline: userMemories,
            compartments,
            memories: trimmed.renderOrder,
            facts: [],
            memoryRenderOptions,
            historyBudgetTokens: budget,
            userProfileBudgetTokens: options.userProfileBudgetTokens,
            decayPressureMultiplier,
        });
        attempts += 1;
    }
    if (m0Text.length === 0) m0Text = M0_EMPTY_BODY;
    return {
        m0Bytes: Buffer.from(m0Text, "utf8"),
        snapshotMarkers,
        renderedMemoryIds: trimmed.renderOrder.map((memory) => memory.id),
    };
}

export function injectM0M1(options: M0M1RenderOptions): InjectM0M1Result {
    if (!options.workspaceIdentitySet && options.projectPath) {
        options = {
            ...options,
            workspaceIdentitySet: resolveWorkspaceIdentitySet(options.db, options.projectPath),
        };
    }
    const skipped: InjectM0M1Result = {
        injected: false,
        m0RematerializedThisPass: false,
        materializationContentionRetryExhausted: false,
        decision: { value: false, reason: "skipped" },
        m0Bytes: options.state.cachedM0Bytes,
        m1Text: null,
    };
    if (options.state.isSubagent) return skipped;

    const decision = mustMaterialize({
        db: options.db,
        sessionId: options.sessionId,
        state: options.state,
        projectPath: options.projectPath,
        projectDirectory: options.projectDirectory,
        hardSignals: options.hardSignals,
        workspaceIdentitySet: options.workspaceIdentitySet,
    });
    let rematerialized = false;
    let contentionExhausted = false;
    let freshFallbackRenderedMemoryIds: number[] | null = null;
    let m1Render: RenderM1Result | null = null;

    if (decision.value) {
        try {
            const materialized = materializeWithRetry(options);
            applyMarkersToState(
                options.state,
                materialized.m0Bytes,
                materialized.snapshotMarkers,
                materialized.m1Bytes,
            );
            m1Render = { text: materialized.m1Text, memoryUpdateCount: 0 };
            rematerialized = true;
        } catch (error) {
            if (!(error instanceof MaterializeContentionError)) throw error;
            if (options.state.cachedM0Bytes && options.state.cachedM1Bytes) {
                // Preferred fallback: reuse the cached baseline. A sibling process
                // mutated state mid-materialization; serving the slightly stale
                // cached m[0]/m[1] pair this pass is correct and the next pass retries.
                // Require BOTH byte buffers: reusing m[0] alone would later hit
                // replayCachedM1 with no m[1] and throw RenderM1InvalidMarkersError
                // (which propagates out and drops injection entirely). The
                // partial-cache state (m[0] set, m[1] null) is reachable after a
                // prior fresh-fallback pass set in-memory m[0] without persisting
                // m[1]; in that case fall through to the fresh-render branch below,
                // which renders a complete m[0]/m[1] pair.
                contentionExhausted = true;
                options.state.snapshotMarkers =
                    options.state.snapshotMarkers ?? snapshotMarkersFromCachedM0(options.state);
                sessionLog(
                    options.sessionId,
                    `m[0] materialization contention exhausted after ${error.retries} retries; reusing cached m[0]/m[1]`,
                );
            } else {
                // No cached baseline to reuse — happens when the cache was cleared
                // THIS pass (history refresh) and then hit contention. Dropping
                // injection would send the model ZERO session history, so render a
                // fresh non-persisted m[0]/m[1] pair as a last resort (mirrors Pi
                // injectM0M1Pi). Not cached because we couldn't win the lock; the
                // next pass re-materializes and persists.
                const fresh = renderFreshM0NonPersisted(options);
                options.state.cachedM0Bytes = fresh.m0Bytes;
                options.state.snapshotMarkers = fresh.snapshotMarkers;
                freshFallbackRenderedMemoryIds = fresh.renderedMemoryIds;
                contentionExhausted = true;
                sessionLog(
                    options.sessionId,
                    `m[0] materialization contention exhausted after ${error.retries} retries with no cached fallback; rendered fresh non-persisted m[0]/m[1]`,
                );
            }
        }
    } else {
        options.state.snapshotMarkers =
            options.state.snapshotMarkers ?? snapshotMarkersFromCachedM0(options.state);
    }

    if (!options.state.cachedM0Bytes || !options.state.snapshotMarkers) {
        throw new RenderM1InvalidMarkersError(options.sessionId);
    }

    let m0Text = decodeM0Bytes(options.state.cachedM0Bytes) ?? M0_EMPTY_BODY;
    let m1Text: string;
    let memoryUpdateCount = 0;
    let m1Recomputed = m1Render !== null;

    if (m1Render) {
        m1Text = m1Render.text;
        memoryUpdateCount = m1Render.memoryUpdateCount;
    } else if (contentionExhausted && freshFallbackRenderedMemoryIds) {
        const freshM1 = renderM1WithMetadata(
            { ...options, preRenderedKeyFilesBlock: preRenderKeyFilesBlock(options) },
            options.state.snapshotMarkers,
            freshFallbackRenderedMemoryIds,
        );
        m1Text = freshM1.text;
        memoryUpdateCount = freshM1.memoryUpdateCount;
        m1Recomputed = true;
    } else if (contentionExhausted) {
        m1Text = replayCachedM1(options.state);
    } else if (options.isCacheBustingPass) {
        const refreshed = softRefreshCachedM1(options);
        m1Text = refreshed.text;
        memoryUpdateCount = refreshed.memoryUpdateCount;
        m1Recomputed = true;
        m0Text = decodeM0Bytes(options.state.cachedM0Bytes) ?? M0_EMPTY_BODY;
    } else {
        m1Text = replayCachedM1(options.state);
    }

    // Pressure backstop refold: the "or we have to due to pressures" half of the
    // m[0]/m[1] contract. When NO HARD bust (TTL/system/tools/model) has arrived
    // but the volatile m[1] delta has grown large, fold it into m[0] (re-run
    // decay, reset m[1]) so a marathon active session can't grow m[1] unbounded.
    // Runs only on cache-busting passes where m[1] was freshly recomputed; defer
    // passes replay persisted bytes and must never live-read/refold.
    //
    // Three independent triggers (any one folds):
    //   1. memoryUpdateCount > 40 — supersede-delta drift (size-independent).
    //   2. m[1]/m[0] SIZE RATIO — m[1] grew past 15% of the m[0] baseline. Gated
    //      by M0_DRIFT_RATIO_FLOOR so a tiny early m[0] (M0_EMPTY_BODY ~35 chars)
    //      doesn't make 15% trivially exceeded and refold every pass.
    //   3. m[1] ABSOLUTE CAP — when m[0] is small the ratio test is suppressed, so
    //      m[1] could otherwise grow without bound. Fold once m[1] alone exceeds a
    //      fixed share of the history budget, independent of m[0] size. estimateTokens
    //      here is fine: this whole branch is rare (cache-busting + m1Recomputed).
    // Small-m[0] floor in TOKENS (not chars): below this the ratio test is
    // suppressed because a small m[0] makes the 15% ratio trivially exceeded.
    const M0_DRIFT_RATIO_FLOOR_TOKENS = 500;
    const M1_DRIFT_RATIO = 0.15;
    const M1_ABSOLUTE_CAP_RATIO = 0.2;
    const m1AbsoluteBudget =
        (options.historyBudgetTokens ?? DEFAULT_HISTORY_BUDGET_TOKENS) * M1_ABSOLUTE_CAP_RATIO;
    // Token counts (NOT char lengths): the documented intent is "m[1] exceeds
    // ~15% of m[0] tokens". XML-heavy / non-Latin content makes char length
    // diverge sharply from token count, so the ratio must compare tokens on both
    // sides. Computed once; this branch is rare (cache-busting + m1Recomputed).
    const m1HasContent = m1Text !== M1_EMPTY_PLACEHOLDER;
    const m1Tokens = m1HasContent ? estimateTokens(m1Text) : 0;
    const m0Tokens = estimateTokens(m0Text);
    const m1OverAbsoluteCap = m1HasContent && m1Tokens > m1AbsoluteBudget;
    if (
        !rematerialized &&
        !contentionExhausted &&
        m1Recomputed &&
        options.isCacheBustingPass &&
        (memoryUpdateCount > 40 ||
            m1OverAbsoluteCap ||
            (m1HasContent &&
                m0Tokens >= M0_DRIFT_RATIO_FLOOR_TOKENS &&
                m1Tokens > m0Tokens * M1_DRIFT_RATIO))
    ) {
        try {
            const refolded = materializeWithRetry(options);
            applyMarkersToState(
                options.state,
                refolded.m0Bytes,
                refolded.snapshotMarkers,
                refolded.m1Bytes,
            );
            rematerialized = true;
            m0Text = decodeM0Bytes(options.state.cachedM0Bytes) ?? M0_EMPTY_BODY;
            m1Text = refolded.m1Text;
        } catch (error) {
            // Contention during the drift refold is non-fatal: keep the current
            // (un-refolded) m[0]/m[1]; the next pass retries the fold.
            if (!(error instanceof MaterializeContentionError)) throw error;
        }
    }

    if (options.messages) {
        prependM0M1Messages(options.sessionId, options.messages, m0Text, m1Text);
    }

    return {
        injected: true,
        m0RematerializedThisPass: rematerialized,
        materializationContentionRetryExhausted: contentionExhausted,
        decision,
        m0Bytes: options.state.cachedM0Bytes,
        m1Text,
    };
}
