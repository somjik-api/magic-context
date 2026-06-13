/**
 * Pi-side wrapper for the `ctx_memory` tool.
 *
 * Action surface mirrors OpenCode's `packages/plugin/src/tools/ctx-memory/tools.ts`.
 * Two tiers of actions:
 *
 *  Primary (for any agent that can call ctx_memory):
 *    - write: insert a new memory (or no-op + bump seenCount on dedup hit)
 *    - archive: soft-delete a memory (status = 'archived'), optional reason
 *    - update: rewrite a memory's content (recomputes normalized_hash + queues re-embed)
 *    - merge: combine N memories into one canonical, supersede the rest
 *
 *  Dreamer-only (gated on `allowDreamerActions: true`):
 *    - list: list active memories for the current project
 *
 * Allowlist gating mirrors OpenCode's `allowedActions` deps field. In OpenCode,
 * the dreamer subagent gets the full action surface because `toolContext.agent
 * === DREAMER_AGENT`. Pi has no agent identity inside child processes, so we
 * use an explicit flag (`--magic-context-dreamer-actions`) wired through the
 * subagent extension entry. Same effective behavior, different transport.
 *
 * Parity reference (OpenCode):
 *   - `tools/ctx-memory/types.ts` for action enum
 *   - `tools/ctx-memory/tools.ts` for handler logic
 *   - `plugin/tool-registry.ts` for the primary allowedActions (CTX_MEMORY_ACTIONS)
 *
 * Memories are project-scoped via `resolveProjectIdentity(ctx.cwd)` and stored
 * in the shared cortexkit DB, so a memory written from the pi-plugin is
 * immediately visible to OpenCode sessions on the same project (and vice
 * versa).
 */

import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import {
	archiveMemory,
	getMemoriesByProject,
	getMemoryByHash,
	getMemoryById,
	insertMemory,
	type Memory,
	type MemoryCategory,
	mergeMemoryStats,
	saveEmbedding,
	supersededMemory,
	updateMemoryContent,
	updateMemorySeenCount,
	V2_MEMORY_CATEGORIES,
} from "@magic-context/core/features/magic-context/memory";
import {
	embedTextForProject,
	getProjectEmbeddingSnapshot,
} from "@magic-context/core/features/magic-context/memory/embedding";
import { computeNormalizedHash } from "@magic-context/core/features/magic-context/memory/normalize-hash";
import {
	normalizeStoredProjectPath,
	resolveProjectIdentity,
	storedPathBelongsToIdentity,
} from "@magic-context/core/features/magic-context/memory/project-identity";
import {
	type ContextDatabase,
	queueMemoryMutation,
} from "@magic-context/core/features/magic-context/storage";
import {
	expandWorkspaceIdentitySetWithAliases,
	resolveStoredPathWorkspaceIdentity,
	resolveWorkspaceIdentitySet,
	resolveWorkspaceShareCategories,
	storedPathBelongsToWorkspace,
} from "@magic-context/core/features/magic-context/workspaces";
import { log } from "@magic-context/core/shared/logger";
import { CTX_MEMORY_DESCRIPTION } from "@magic-context/core/tools/ctx-memory/constants";
import { type Static, Type } from "typebox";

// Default page size for `list`. Larger than the search default because
// maintenance tasks page through the full memory set, but still bounded.
const DEFAULT_LIST_LIMIT = 100;
// Hard char budget for a single `list` page. A project can accumulate
// thousands of active memories (hundreds of KB); returning them all in one
// tool result overflows the dreamer model context and deadlocks the
// consolidate/verify/archive tasks. The handler stops adding rows at this
// budget and points the caller at the next page via `offset`.
const LIST_PAGE_CHAR_BUDGET = 24000;

// Mirrors OpenCode CTX_MEMORY_DREAMER_ACTIONS. `delete` was removed — it was an
// exact alias of `archive` (both soft-archive); `archive` is the single
// soft-remove action. Primary agents get write/archive/update/merge on the
// memories they already see (with ids) in the injected project-memory block;
// only `list` (bulk enumeration) stays dreamer-only.
const ALL_ACTIONS = ["write", "archive", "update", "merge", "list"] as const;
type CtxMemoryAction = (typeof ALL_ACTIONS)[number];

const DREAMER_ONLY_ACTIONS: ReadonlySet<CtxMemoryAction> = new Set(["list"]);

const ParamsSchema = Type.Object({
	action: Type.Union(
		ALL_ACTIONS.map((a) => Type.Literal(a)),
		{ description: "What to do: write, update, archive, or merge" },
	),
	content: Type.Optional(
		Type.String({
			description:
				"The memory text — one standalone fact (required for write, update, merge)",
		}),
	),
	category: Type.Optional(
		Type.Union(
			V2_MEMORY_CATEGORIES.map((c) => Type.Literal(c)),
			{
				description:
					"What kind of fact this is (required for write; optional merge override)",
			},
		),
	),
	ids: Type.Optional(
		Type.Array(Type.Number(), {
			description:
				"Target memory id(s) from <project-memory>: update takes exactly one, archive one or more, merge two or more",
		}),
	),
	limit: Type.Optional(
		Type.Number({
			description: "Maximum results to return for list (default: 100)",
		}),
	),
	offset: Type.Optional(
		Type.Number({
			description: "Zero-based offset for list pagination (default: 0)",
		}),
	),
	reason: Type.Optional(
		Type.String({
			description: "Why the memory is being archived (optional, recommended)",
		}),
	),
});

type CtxMemoryParams = Static<typeof ParamsSchema>;

function ok(text: string) {
	return { content: [{ type: "text" as const, text }], details: undefined };
}

function err(text: string) {
	return {
		content: [{ type: "text" as const, text }],
		details: undefined,
		isError: true,
	};
}

function normalizeLimit(
	limit?: number,
	fallback: number = DEFAULT_LIST_LIMIT,
): number {
	if (typeof limit !== "number" || !Number.isFinite(limit)) return fallback;
	return Math.max(1, Math.floor(limit));
}

function normalizeOffset(offset?: number): number {
	if (typeof offset !== "number" || !Number.isFinite(offset)) return 0;
	return Math.max(0, Math.floor(offset));
}

interface MemoryPage {
	pageMemories: Memory[];
	totalCount: number;
	offset: number;
}

function formatMemoryList(page: MemoryPage): string {
	const { pageMemories, totalCount, offset } = page;
	if (totalCount === 0) return "No active memories found.";
	if (pageMemories.length === 0)
		return `No memories at offset ${offset}. Total is ${totalCount}; use a smaller offset.`;

	const allRows = pageMemories.map((m) => ({
		id: String(m.id),
		category: m.category,
		status: m.status,
		updated: new Date(m.updatedAt).toISOString(),
		content: m.content.replace(/\s+/g, " ").trim(),
	}));

	// Hard char budget so a single page can never overflow the model context,
	// regardless of how large `limit` is. Always include at least one row.
	const rows: typeof allRows = [];
	let usedChars = 0;
	for (const r of allRows) {
		const rowChars = r.content.length + 80;
		if (rows.length > 0 && usedChars + rowChars > LIST_PAGE_CHAR_BUDGET) break;
		rows.push(r);
		usedChars += rowChars;
	}

	const widths = {
		id: Math.max(2, ...rows.map((r) => r.id.length)),
		category: Math.max(8, ...rows.map((r) => r.category.length)),
		status: Math.max(6, ...rows.map((r) => r.status.length)),
		updated: Math.max(7, ...rows.map((r) => r.updated.length)),
	};
	const fmt = (r: (typeof rows)[number]) =>
		[
			r.id.padEnd(widths.id),
			r.category.padEnd(widths.category),
			r.status.padEnd(widths.status),
			r.updated.padEnd(widths.updated),
			r.content,
		].join(" | ");

	const shownEnd = offset + rows.length;
	const hasMore = shownEnd < totalCount;
	const footer = hasMore
		? `\n\n… ${totalCount - shownEnd} more. Fetch the next page with ctx_memory(action="list", offset=${shownEnd}${
				pageMemories.length > rows.length ? "" : `, limit=${rows.length}`
			}).`
		: "";

	return [
		`Showing memories ${offset + 1}-${shownEnd} of ${totalCount} total.`,
		"",
		...rows.map(fmt),
	]
		.join("\n")
		.concat(footer);
}

function isPrimaryMutableMemory(memory: Memory): boolean {
	return (
		(memory.status === "active" || memory.status === "permanent") &&
		memory.supersededByMemoryId === null
	);
}

function inactiveMemoryError(
	id: number,
	action: "updating" | "merging" | "archiving",
): string {
	return `Error: Memory with ID ${id} is archived or superseded; restore it before ${action}.`;
}

function queueEmbedding(args: {
	deps: CtxMemoryToolDeps;
	projectIdentity: string;
	memoryId: number;
	content: string;
}) {
	const snapshot = getProjectEmbeddingSnapshot(args.projectIdentity);
	if (!snapshot?.enabled) return;
	void (async () => {
		try {
			const result = await embedTextForProject(
				args.projectIdentity,
				args.content,
			);
			if (!result) {
				log(
					`[magic-context-pi] embedding skipped for memory ${args.memoryId}: provider unavailable.`,
				);
				return;
			}
			saveEmbedding(args.deps.db, args.memoryId, result.vector, result.modelId);
			log(`[magic-context-pi] proactively embedded memory ${args.memoryId}.`);
		} catch (error) {
			log(
				`[magic-context-pi] embedding failed for memory ${args.memoryId}:`,
				error,
			);
		}
	})();
}

export interface CtxMemoryToolDeps {
	db: ContextDatabase;
	ensureProjectRegistered?: (
		directory: string,
		db: ContextDatabase,
	) => Promise<void>;
	memoryEnabled?: boolean;
	embeddingEnabled?: boolean;
	/** When true, the dreamer-only `list` action is exposed. Set by the subagent
	 *  extension entry when the parent passes `--magic-context-dreamer-actions`.
	 *  Default: false (primary set only: write/archive/update/merge). */
	allowDreamerActions?: boolean;
}

export function createCtxMemoryTool(
	deps: CtxMemoryToolDeps,
): ToolDefinition<typeof ParamsSchema> {
	const dreamerAllowed = deps.allowDreamerActions === true;
	const description = dreamerAllowed
		? `${CTX_MEMORY_DESCRIPTION}\n- list: enumerate stored memories (maintenance sessions).`
		: CTX_MEMORY_DESCRIPTION;

	return {
		name: "ctx_memory",
		label: "Magic Context: Memory",
		description,
		parameters: ParamsSchema,
		async execute(
			_toolCallId,
			params: CtxMemoryParams,
			_signal,
			_onUpdate,
			ctx,
		) {
			// Gate dreamer-only actions on the allowlist flag. Mirrors
			// OpenCode's `if (toolContext.agent !== DREAMER_AGENT && !allowedActions.includes(args.action))`.
			if (!dreamerAllowed && DREAMER_ONLY_ACTIONS.has(params.action)) {
				return err(
					`Error: Action '${params.action}' is not allowed in this context.`,
				);
			}

			const projectIdentity = resolveProjectIdentity(ctx.cwd);
			await deps.ensureProjectRegistered?.(ctx.cwd, deps.db);
			const workspaceIdentitySet = resolveWorkspaceIdentitySet(
				deps.db,
				projectIdentity,
			);
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
						) ?? normalizeStoredProjectPath(rawProjectPath))
					: normalizeStoredProjectPath(rawProjectPath);
			// The workspace's share-category policy, identical to the render path
			// (resolveWorkspaceRenderContextPi): null = share all categories.
			const toolShareCategories =
				workspaceIdentitySet.identities.length > 1
					? resolveWorkspaceShareCategories(deps.db, projectIdentity)
					: null;
			// Tool visibility MUST match render visibility, or the agent could
			// mutate (update/archive) a foreign workspace memory it can't even
			// see. Own-project memories: every category is mutable. Foreign member
			// memories: only when shared — shareCategories===null shares all, an
			// empty list shares none, otherwise only the listed categories.
			// Mirrors buildWorkspaceMemorySqlFilter's own/foreign split.
			const memoryVisibleToTool = (memory: Memory): boolean => {
				if (workspaceIdentitySet.identities.length <= 1) {
					return storedPathBelongsToIdentity(
						memory.projectPath,
						projectIdentity,
					);
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
				const isOwn =
					targetIdentityForStoredPath(memory.projectPath) === projectIdentity;
				if (isOwn) return true;
				return (
					toolShareCategories === null ||
					toolShareCategories.includes(memory.category)
				);
			};
			const snapshot = getProjectEmbeddingSnapshot(projectIdentity);
			if (
				snapshot
					? !snapshot.features.memoryEnabled
					: deps.memoryEnabled === false
			) {
				return err("Cross-session memory is disabled for this project.");
			}
			const sessionId = ctx.sessionManager.getSessionId();

			if (params.action === "write") {
				const content = params.content?.trim();
				if (!content)
					return err("Error: 'content' is required when action is 'write'.");

				const rawCategory = params.category;
				if (!rawCategory) {
					return err("Error: 'category' is required when action is 'write'.");
				}

				const existing = getMemoryByHash(
					deps.db,
					projectIdentity,
					rawCategory,
					computeNormalizedHash(content),
				);
				if (existing) {
					updateMemorySeenCount(deps.db, existing.id);
					return ok(
						`Memory already exists [ID: ${existing.id}] in ${rawCategory} (seen count incremented).`,
					);
				}

				const memory = insertMemory(deps.db, {
					projectPath: projectIdentity,
					category: rawCategory,
					content,
					sourceSessionId: sessionId,
					sourceType: dreamerAllowed ? "dreamer" : "agent",
				});

				queueEmbedding({ deps, projectIdentity, memoryId: memory.id, content });
				// Do NOT invalidate the m[0]/m[1] cache here. An additive write is a
				// supersede-delta operation: it surfaces in m[1] via the maxMemoryId
				// watermark (renderM1 reads memories with id > cachedM0MaxMemoryId) on
				// the next cache-busting pass, WITHOUT busting m[0]. Clearing the cache
				// re-materialized m[0] for every session (the call was global over
				// session_meta), defeating the whole additive/non-additive split and
				// busting unrelated projects. Matches OpenCode's write path, which
				// likewise does no cache invalidation.
				return ok(`Saved memory [ID: ${memory.id}] in ${rawCategory}.`);
			}

			if (params.action === "list") {
				const limit = normalizeLimit(params.limit, DEFAULT_LIST_LIMIT);
				const offset = normalizeOffset(params.offset);
				const filtered = getMemoriesByProject(deps.db, projectIdentity);
				const category = params.category;
				const filtered2 = category
					? filtered.filter((m) => m.category === category)
					: filtered;
				const pageMemories = filtered2.slice(offset, offset + limit);
				return ok(
					formatMemoryList({
						pageMemories,
						totalCount: filtered2.length,
						offset,
					}),
				);
			}

			if (params.action === "update") {
				const updateIds = params.ids;
				if (
					!updateIds ||
					updateIds.length !== 1 ||
					!updateIds.every(Number.isInteger)
				) {
					return err(
						"Error: 'ids' must contain exactly one integer memory ID when action is 'update'.",
					);
				}
				const updateId = updateIds[0];
				const content = params.content?.trim();
				if (!content) {
					return err("Error: 'content' is required when action is 'update'.");
				}

				const memory = getMemoryById(deps.db, updateId);
				if (!memory || !memoryVisibleToTool(memory)) {
					return err(`Error: Memory with ID ${updateId} was not found.`);
				}
				if (!dreamerAllowed && !isPrimaryMutableMemory(memory)) {
					return err(inactiveMemoryError(updateId, "updating"));
				}

				const normalizedHash = computeNormalizedHash(content);
				const targetIdentity = targetIdentityForStoredPath(memory.projectPath);
				const duplicate = getMemoryByHash(
					deps.db,
					targetIdentity,
					memory.category,
					normalizedHash,
				);
				if (duplicate && duplicate.id !== memory.id) {
					return err(
						`Error: Memory content already exists as ID ${duplicate.id}; merge or archive duplicates instead.`,
					);
				}

				deps.db.transaction(() => {
					updateMemoryContent(deps.db, memory.id, content, normalizedHash);
					queueMemoryMutation(deps.db, {
						projectPath: targetIdentity,
						mutationType: "update",
						targetMemoryId: memory.id,
						category: memory.category,
						newContent: content,
					});
				})();
				queueEmbedding({
					deps,
					projectIdentity: targetIdentity,
					memoryId: memory.id,
					content,
				});
				return ok(`Updated memory [ID: ${memory.id}] in ${memory.category}.`);
			}

			if (params.action === "merge") {
				const ids = params.ids;
				if (!ids || ids.length < 2 || !ids.every(Number.isInteger)) {
					return err(
						"Error: 'ids' must include at least two integer memory IDs when action is 'merge'.",
					);
				}
				if (new Set(ids).size !== ids.length) {
					return err(
						"Error: 'ids' must include at least two distinct memory IDs when action is 'merge'.",
					);
				}

				const content = params.content?.trim();
				if (!content) {
					return err("Error: 'content' is required when action is 'merge'.");
				}

				const sourceMemories = ids
					.map((id) => getMemoryById(deps.db, id))
					.filter((memory): memory is Memory => Boolean(memory));
				if (sourceMemories.length !== ids.length) {
					return err("Error: One or more source memories were not found.");
				}

				// Cross-identity consolidation is a DREAMER-ONLY capability: each
				// source is superseded under ITS OWN project identity with a
				// per-project supersede-delta row (see the supersede loop below),
				// so every affected project's m[1] reconciles correctly. But
				// `merge` is in the primary action set too, and a primary agent
				// must not reach into ANOTHER project's memories — mirror
				// update/archive ownership (parity with OpenCode).
				if (!dreamerAllowed) {
					const foreign = sourceMemories.find(
						(memory) => !memoryVisibleToTool(memory),
					);
					if (foreign) {
						return err(`Error: Memory with ID ${foreign.id} was not found.`);
					}
					const inactive = sourceMemories.find(
						(memory) => !isPrimaryMutableMemory(memory),
					);
					if (inactive) {
						return err(inactiveMemoryError(inactive.id, "merging"));
					}
				}

				// Schema-validated literal union — no runtime re-check needed.
				const requestedCategoryTyped: MemoryCategory | undefined =
					params.category;
				const fallbackCategory = sourceMemories[0]?.category;
				const category: MemoryCategory | undefined =
					requestedCategoryTyped ?? fallbackCategory;
				if (!category) {
					return err(
						"Error: A valid category is required when action is 'merge'.",
					);
				}

				if (
					!requestedCategoryTyped &&
					sourceMemories.some((memory) => memory.category !== category)
				) {
					return err(
						"Error: Mixed-category merges require an explicit 'category'.",
					);
				}

				const normalizedHash = computeNormalizedHash(content);
				const duplicate = getMemoryByHash(
					deps.db,
					projectIdentity,
					category,
					normalizedHash,
				);
				const canonicalExisting =
					duplicate && ids.includes(duplicate.id) ? duplicate : null;
				if (duplicate && !canonicalExisting) {
					return err(
						`Error: Memory content already exists as ID ${duplicate.id}; update or archive existing duplicates instead.`,
					);
				}

				// Aggregate stats from all source memories.
				const mergedSeenCount = sourceMemories.reduce(
					(sum, memory) => sum + memory.seenCount,
					0,
				);
				const mergedRetrievalCount = sourceMemories.reduce(
					(sum, memory) => sum + memory.retrievalCount,
					0,
				);
				// `mergedFrom` is JSON-stringified in the DB. Flatten any prior
				// merge chains so the lineage stays accurate when merging
				// already-merged memories. Mirrors OpenCode's parity construction
				// at packages/plugin/src/tools/ctx-memory/tools.ts:381-405.
				const mergedFromIds = Array.from(
					new Set(
						sourceMemories.flatMap((memory) => {
							let parsed: unknown[] = [];
							try {
								parsed = memory.mergedFrom ? JSON.parse(memory.mergedFrom) : [];
							} catch {
								parsed = [];
							}
							const priorIds = Array.isArray(parsed)
								? parsed.filter(
										(value): value is number => typeof value === "number",
									)
								: [];
							return [memory.id, ...priorIds];
						}),
					),
				).sort((left, right) => left - right);
				const mergedFrom = JSON.stringify(mergedFromIds);
				const mergedStatus: "active" | "permanent" = sourceMemories.some(
					(memory) => memory.status === "permanent",
				)
					? "permanent"
					: "active";

				let canonicalMemory!: Memory;
				deps.db.transaction(() => {
					let canonicalContentChanged = false;
					if (canonicalExisting) {
						// One of the source memories already has the merged content.
						// Update it in place to absorb stats from the others.
						canonicalMemory = canonicalExisting;
						canonicalContentChanged =
							canonicalMemory.content !== content ||
							canonicalMemory.normalizedHash !== normalizedHash;
						if (canonicalContentChanged) {
							updateMemoryContent(
								deps.db,
								canonicalMemory.id,
								content,
								normalizedHash,
							);
						}
					} else {
						// Insert a fresh canonical memory with the merged content.
						canonicalMemory = insertMemory(deps.db, {
							projectPath: projectIdentity,
							category,
							content,
							sourceSessionId: sessionId,
							sourceType: dreamerAllowed ? "dreamer" : "agent",
						});
					}

					mergeMemoryStats(
						deps.db,
						canonicalMemory.id,
						mergedSeenCount,
						mergedRetrievalCount,
						mergedFrom,
						mergedStatus,
					);

					for (const memory of sourceMemories) {
						if (memory.id === canonicalMemory.id) {
							continue;
						}
						supersededMemory(deps.db, memory.id, canonicalMemory.id);
						queueMemoryMutation(deps.db, {
							// Normalize the stored path to the resolved identity
							// before queueing — the render-side mutation-log reader
							// matches exact project_path, and OpenCode + dashboard
							// both normalize first. A legacy raw filesystem path here
							// would write a row that normalized git:/dir: sessions
							// never read (the supersede delta would silently vanish).
							projectPath: normalizeStoredProjectPath(memory.projectPath),
							mutationType: "superseded",
							targetMemoryId: memory.id,
							supersededById: canonicalMemory.id,
						});
					}

					if (canonicalExisting && canonicalContentChanged) {
						queueMemoryMutation(deps.db, {
							projectPath: normalizeStoredProjectPath(
								canonicalMemory.projectPath,
							),
							mutationType: "update",
							targetMemoryId: canonicalMemory.id,
							category,
							newContent: content,
						});
					}
				})();

				queueEmbedding({
					deps,
					projectIdentity,
					memoryId: canonicalMemory.id,
					content,
				});
				const supersededIds = sourceMemories
					.map((memory) => memory.id)
					.filter((id) => id !== canonicalMemory.id);
				return ok(
					`Merged memories [${ids.join(", ")}] into canonical memory [ID: ${canonicalMemory.id}] in ${category}; superseded [${supersededIds.join(", ")}].`,
				);
			}

			if (params.action === "archive") {
				const rawArchiveIds = params.ids;
				if (
					!rawArchiveIds ||
					rawArchiveIds.length === 0 ||
					!rawArchiveIds.every(Number.isInteger)
				) {
					return err(
						"Error: 'ids' must contain at least one integer memory ID when action is 'archive'.",
					);
				}
				// De-dupe (first-seen order): `ids:[42,42]` archives once.
				const archiveIds = [...new Set(rawArchiveIds)];
				// Validate the whole batch BEFORE mutating so a typo'd id can't
				// half-archive a batch (all-or-nothing, matching the transaction).
				for (const memoryId of archiveIds) {
					const memory = getMemoryById(deps.db, memoryId);
					if (!memory || !memoryVisibleToTool(memory)) {
						return err(`Error: Memory with ID ${memoryId} was not found.`);
					}
					if (!dreamerAllowed && !isPrimaryMutableMemory(memory)) {
						// Match update/merge: once a primary caller archived or
						// superseded a memory, re-archiving it should stop with the same
						// friendly inactive-memory error instead of rewriting it.
						return err(inactiveMemoryError(memoryId, "archiving"));
					}
				}
				const targets = archiveIds.map((memoryId) => {
					const memory = getMemoryById(deps.db, memoryId);
					if (!memory)
						throw new Error(`validated memory ${memoryId} disappeared`);
					return {
						memoryId,
						projectIdentity: targetIdentityForStoredPath(memory.projectPath),
					};
				});
				deps.db.transaction(() => {
					for (const target of targets) {
						archiveMemory(deps.db, target.memoryId, params.reason);
						queueMemoryMutation(deps.db, {
							projectPath: target.projectIdentity,
							mutationType: "archive",
							targetMemoryId: target.memoryId,
						});
					}
				})();
				const reasonSuffix = params.reason ? ` (${params.reason})` : "";
				const idList = archiveIds.join(", ");
				const plural = archiveIds.length > 1 ? "memories" : "memory";
				return ok(`Archived ${plural} [ID: ${idList}]${reasonSuffix}.`);
			}

			return err("Error: Unknown action.");
		},
	};
}
