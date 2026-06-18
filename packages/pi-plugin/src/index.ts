/**
 * Magic Context — Pi coding agent extension.
 *
 * Loaded once per Pi session via `pi.extensions` in package.json. Boots
 * Magic Context's shared SQLite store and registers session lifecycle
 * hooks: tools, transform pipeline (tagging + drops), historian trigger,
 * /ctx-aug command, system-prompt injection, dreamer scheduling, and
 * agent_end cleanup.
 *
 * Storage: shares one SQLite database with the OpenCode plugin at
 *   ~/.local/share/cortexkit/magic-context/context.db
 * so project memories, embedding cache, dreamer runs, and other
 * project-scoped state are visible across both harnesses. Session-scoped
 * tables carry a `harness` column ('opencode' or 'pi') so per-session
 * data stays correctly attributed.
 *
 * Config: read from the shared CortexKit location —
 *   $cwd/.cortexkit/magic-context.jsonc (project) and
 *   ~/.config/cortexkit/magic-context.jsonc (user) via `loadPiConfig()`.
 *   Falls back to schema defaults when neither file exists.
 */

import { createRequire } from "node:module";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { isDreamerRunnable } from "@magic-context/core/config/agent-disable";
import { migrateMagicContextConfigLocations } from "@magic-context/core/config/migrate-config-location";
import type {
	DreamerConfig,
	HistorianConfig,
	MagicContextConfig,
	SidekickConfig,
} from "@magic-context/core/config/schema/magic-context";
import {
	summarizeDreamSchedule,
	userMemoryCollectionEnabled,
} from "@magic-context/core/features/magic-context/dreamer/task-config";
import { resolveProjectIdentity } from "@magic-context/core/features/magic-context/memory/project-identity";
import { scheduleIncrementalIndex } from "@magic-context/core/features/magic-context/message-index-async";
import { detectOverflow } from "@magic-context/core/features/magic-context/overflow-detection";
import type { ContextDatabase } from "@magic-context/core/features/magic-context/storage";
import {
	getOrCreateSessionMeta,
	getPendingPiCompactionMarkerState,
	getSessionsWithPendingPiMarker,
	updateSessionMeta,
} from "@magic-context/core/features/magic-context/storage";
import {
	applySqliteTuningPragmas,
	openDatabase,
	setSqlitePragmaConfig,
} from "@magic-context/core/features/magic-context/storage-db";
import {
	getOverflowState,
	recordOverflowDetected,
} from "@magic-context/core/features/magic-context/storage-meta-persisted";
import { runDeferredV22Backfill } from "@magic-context/core/features/magic-context/v22-deferred-backfill";
import {
	deriveHistorianChunkTokens,
	resolveHistorianContextLimit,
} from "@magic-context/core/hooks/magic-context/derive-budgets";
import { resolveCacheTtl } from "@magic-context/core/hooks/magic-context/event-resolvers";
import {
	clearNoteNudgeTriggerAndCooldown,
	onNoteTrigger,
} from "@magic-context/core/hooks/magic-context/note-nudger";
import { normalizeTodoStateJson } from "@magic-context/core/hooks/magic-context/todo-view";
import { maybeSendUpgradeReminder } from "@magic-context/core/hooks/magic-context/upgrade-reminder";
import {
	ANNOUNCEMENT_FEATURES,
	ANNOUNCEMENT_FOOTER,
	ANNOUNCEMENT_VERSION,
	markAnnouncementSeen,
	shouldShowAnnouncement,
} from "@magic-context/core/shared/announcement";
import { getMagicContextStorageDir } from "@magic-context/core/shared/data-path";
import { setHarness } from "@magic-context/core/shared/harness";
import { setKeepSubagents } from "@magic-context/core/shared/keep-subagents";
import { log } from "@magic-context/core/shared/logger";
import { isSaneLimit } from "@magic-context/core/shared/models-dev-cache";
import { resolveFallbackChain } from "@magic-context/core/shared/resolve-fallbacks";
import {
	type PiSidekickConfig,
	registerCtxAugCommand,
} from "./commands/ctx-aug";
import { registerCtxDreamCommand } from "./commands/ctx-dream";
import {
	maybeAutoEmbedPiSession,
	registerCtxEmbedCommand,
} from "./commands/ctx-embed";
import { registerCtxFlushCommand } from "./commands/ctx-flush";
import { registerCtxRecompCommand } from "./commands/ctx-recomp";
import { registerCtxSessionUpgradeCommand } from "./commands/ctx-session-upgrade";
import { registerCtxStatusCommand } from "./commands/ctx-status";
import { loadPiConfig } from "./config";
import {
	awaitInFlightHistorians,
	clearContextHandlerSession,
	clearPiM0Cache,
	clearSystemPromptRefresh,
	hasSystemPromptRefresh,
	type PiAutoSearchHandlerOptions,
	type PiContextHandlerOptions,
	type PiHistorianOptions,
	recordPiLiveModel,
	registerPiContextHandler,
	signalPiDeferredHistoryRefresh,
	signalPiHistoryRefresh,
	signalPiPendingMaterialization,
	signalPiSystemPromptRefresh,
	signalPiSystemPromptRefreshForProject,
	trackSessionForProject,
} from "./context-handler";
import {
	markPiChannel1Reduced,
	maybeChannel1ReminderForToolResult,
	maybeDeliverChannel2Pi,
} from "./ctx-reduce-nudge-pi";
import {
	awaitInFlightDreamers,
	registerPiDreamerProject,
	unregisterPiDreamerProject,
} from "./dreamer";
import { ensureProjectRegisteredFromPiDirectory } from "./embedding-bootstrap";
import { computePiPressure, extractAssistantUsage } from "./pi-pressure";
import { awaitInFlightRecomps } from "./pi-recomp-runner";
import { readPiSessionMessages } from "./read-session-pi";
import { registerStatusLine, updateStatusLine } from "./status-line";
import { stripTagPrefixFromAssistantMessage } from "./strip-tag-prefix";
import magicContextSubagentExtension from "./subagent-entry";
import {
	MAGIC_CONTEXT_PI_SUBAGENT_ENV,
	PiSubagentRunner,
} from "./subagent-runner";
import {
	buildMagicContextBlock,
	clearPiSystemPromptSession,
	processSystemPromptForCache,
} from "./system-prompt";
import { withTimeout } from "./timeout";
import { registerMagicContextTools } from "./tools";

const PREFIX = "[magic-context][pi]";

function resolveCurrentProject(ctx: { cwd: string }): {
	projectDir: string;
	projectIdentity: string;
} {
	const projectDir = ctx.cwd;
	const projectIdentity = resolveProjectIdentity(projectDir);
	return { projectDir, projectIdentity };
}

export function persistPiMessageEndModelMeta(args: {
	db: ContextDatabase;
	sessionId: string;
	message: unknown;
	cacheTtlConfig: MagicContextConfig["cache_ttl"];
}): void {
	if (!args.message || typeof args.message !== "object") return;
	const msg = args.message as {
		role?: string;
		provider?: string;
		model?: string;
	};
	if (
		msg.role !== "assistant" ||
		typeof msg.provider !== "string" ||
		msg.provider.length === 0 ||
		typeof msg.model !== "string" ||
		msg.model.length === 0
	) {
		return;
	}
	const modelKey = `${msg.provider}/${msg.model}`;
	recordPiLiveModel(args.sessionId, modelKey);
	const cacheTtl = resolveCacheTtl(args.cacheTtlConfig, modelKey);
	const currentMeta = getOrCreateSessionMeta(args.db, args.sessionId);
	if (currentMeta.cacheTtl !== cacheTtl) {
		updateSessionMeta(args.db, args.sessionId, { cacheTtl });
	}
}

function info(message: string, data?: unknown): void {
	log(`${PREFIX} ${message}`, data);
}

function warn(message: string, data?: unknown): void {
	log(`${PREFIX} WARN ${message}`, data);
}

// Migrate config from the legacy per-harness locations to the shared CortexKit
// location BEFORE any loadPiConfig (hard cutover: the loader reads only
// CortexKit). Memoized per directory so the per-cwd switch sites don't re-run
// the (idempotent, lock-guarded) migration on every pass. Fails open.
const migratedConfigDirs = new Set<string>();
function ensureConfigLocationsMigrated(dir: string): void {
	if (migratedConfigDirs.has(dir)) return;
	migratedConfigDirs.add(dir);
	migrateMagicContextConfigLocations(dir, {
		warn: (m) => warn(m),
		info: (m) => info(m),
	});
}

function formatTokens(value: number): string {
	return value.toLocaleString();
}

function getPiMessageModel(message: unknown): {
	provider: string | undefined;
	model: string | undefined;
} {
	if (!message || typeof message !== "object") {
		return { provider: undefined, model: undefined };
	}
	const msg = message as { provider?: unknown; model?: unknown };
	return {
		provider: typeof msg.provider === "string" ? msg.provider : undefined,
		model: typeof msg.model === "string" ? msg.model : undefined,
	};
}

function resolvePiPressureContextLimit(args: {
	db: ContextDatabase;
	sessionId: string;
	piContextWindow: number;
}): number {
	// Pi reports the model's context window directly (ctx.getContextUsage() /
	// ctx.getModel().contextWindow) — its own authoritative source. We no longer
	// consult models.dev for Pi. Sanity-bound the reported value so a transient
	// garbage window can't poison pressure (mirrors OpenCode's SDK sane bound).
	let effectiveContextLimit = isSaneLimit(args.piContextWindow)
		? args.piContextWindow
		: 0;
	try {
		const overflowState = getOverflowState(args.db, args.sessionId);
		if (overflowState.detectedContextLimit > 0) {
			effectiveContextLimit =
				effectiveContextLimit > 0
					? Math.min(effectiveContextLimit, overflowState.detectedContextLimit)
					: overflowState.detectedContextLimit;
		}
	} catch (err) {
		warn("message_end: getOverflowState failed:", err);
	}
	return effectiveContextLimit;
}

export async function persistPiPressureFromMessageEnd(args: {
	db: ContextDatabase;
	sessionId: string;
	message: unknown;
	piContextWindow: number;
	piTokens?: number;
	notifyIssue?: (message: string) => unknown | Promise<unknown>;
}): Promise<void> {
	const { provider, model } = getPiMessageModel(args.message);
	const effectiveContextLimit = resolvePiPressureContextLimit({
		db: args.db,
		sessionId: args.sessionId,
		piContextWindow: args.piContextWindow,
	});
	const usage = extractAssistantUsage(args.message);
	const pressure = computePiPressure(usage, effectiveContextLimit);
	const msg =
		args.message && typeof args.message === "object"
			? (args.message as { errorMessage?: unknown })
			: undefined;
	const messageHadOverflowError =
		typeof msg?.errorMessage === "string" &&
		detectOverflow(msg.errorMessage).isOverflow;
	const updates: Partial<{
		lastResponseTime: number;
		lastContextPercentage: number;
		lastInputTokens: number;
		observedSafeInputTokens: number;
		cacheAlertSent: boolean;
	}> = { lastResponseTime: Date.now() };

	if (pressure) {
		const percentage = pressure.percentage;
		const contextLimit = effectiveContextLimit;
		const meta = getOrCreateSessionMeta(args.db, args.sessionId);
		const observedSafeInputTokens = meta.observedSafeInputTokens ?? 0;
		if (
			percentage > 100 &&
			observedSafeInputTokens > 0 &&
			pressure.inputTokens <= observedSafeInputTokens * 2
		) {
			// Pi resolves the window from its own runtime, not a cache we could
			// reload — so a >100% reading with a known-good safe baseline means
			// Pi's reported contextWindow is genuinely wrong. There's nothing to
			// re-fetch; surface the alert (overflow detection still captures a
			// real lower cap separately).
			if (!meta.cacheAlertSent) {
				updates.cacheAlertSent = true;
				const safeTokens = Math.max(
					observedSafeInputTokens,
					pressure.inputTokens,
				);
				const modelLabel =
					provider && model ? `${provider}/${model}` : "the active model";
				await args.notifyIssue?.(
					`⚠️ Magic Context: Pi reports a context limit of ${formatTokens(contextLimit)} tokens for ${modelLabel} but you've successfully sent ${formatTokens(safeTokens)} tokens in this session — the reported limit looks wrong. Restart Pi if you suspect this is incorrect.`,
				);
			}
		}
		updates.lastContextPercentage = percentage;
		updates.lastInputTokens = pressure.inputTokens;
		if (!messageHadOverflowError) {
			updates.observedSafeInputTokens = Math.max(
				observedSafeInputTokens,
				pressure.inputTokens,
			);
		}
	} else if (typeof args.piTokens === "number") {
		updates.lastInputTokens = args.piTokens;
		if (args.piContextWindow > 0) {
			updates.lastContextPercentage =
				(args.piTokens / args.piContextWindow) * 100;
		}
	}

	updateSessionMeta(args.db, args.sessionId, updates);
}

/** Plugin version from package.json. */
const PLUGIN_VERSION: string = (() => {
	try {
		const req = createRequire(import.meta.url);
		return (req("../package.json") as { version: string }).version;
	} catch {
		return "0.0.0";
	}
})();

/** Lock the harness at module load. Safe to import this file in tests; the
 * lock is idempotent and will throw only on a conflicting reset. */
setHarness("pi");

// ---------------------------------------------------------------------------
// Config-driven resolvers
//
// Step 5b replaced the env-var stop-gaps with `loadPiConfig()` which reads
// $cwd/.pi/magic-context.jsonc (project) + ~/.pi/agent/magic-context.jsonc
// (user) and merges them through the shared Zod schema. The resolvers below
// adapt the schema-shaped config into the Pi-specific options the various
// registration helpers expect.
//
// Each resolver returns `undefined` when the relevant feature is disabled
// in config, so the registration helpers can short-circuit cleanly.
// ---------------------------------------------------------------------------

export function resolveSidekickFromConfig(
	config: MagicContextConfig,
): PiSidekickConfig | undefined {
	const sidekick = config.sidekick as SidekickConfig | undefined;
	if (!sidekick || sidekick.disable === true) return undefined;
	const model = sidekick.model?.trim();
	if (!model || model.length === 0) return undefined;
	return {
		model,
		systemPrompt: sidekick.system_prompt,
		timeoutMs: sidekick.timeout_ms,
		thinking_level: sidekick.thinking_level,
		fallbackModels: resolveFallbackChain(sidekick.fallback_models),
		language: config.language,
	};
}

export function resolveHistorianFromConfig(
	config: MagicContextConfig,
): PiHistorianOptions | undefined {
	// Defensive: schema declares `historian` required with default {}, but the
	// runtime config can come from a malformed JSONC merge that drops the
	// field. Fall back to undefined-safe access so plugin load never crashes.
	const historian = config.historian as HistorianConfig | undefined;
	if (historian?.disable === true) return undefined;
	const model = historian?.model?.trim();
	if (!model || model.length === 0) return undefined;

	// The historian chunk budget is anchored to the HISTORIAN model because
	// it bounds one summarizer call. The trigger budget is intentionally NOT
	// derived at startup: Pi resolves it per context pass from the live main
	// session model + effective execute threshold to match OpenCode.
	const historianContextLimit = resolveHistorianContextLimit(model);
	const historianChunkTokens = deriveHistorianChunkTokens(
		historianContextLimit,
	);

	const fallbackModels = resolveFallbackChain(historian?.fallback_models);

	return {
		runner: new PiSubagentRunner(),
		model,
		fallbackModels,
		historianChunkTokens,
		timeoutMs: config.historian_timeout_ms,
		// `historian.two_pass` runs an editor pass after a successful
		// first pass to clean low-signal U: lines and cross-compartment
		// duplicates. Mirrors OpenCode's config flag — defaults to false
		// on the schema side because the editor pass adds a second
		// historian round-trip's latency and token cost. Enable for
		// long sessions where chunk dedupe matters more than speed.
		twoPass: historian?.two_pass === true,
		// Pi only: explicit thinking level for historian subagent invocations.
		// When set, passed as --thinking <level> to Pi subprocess.
		// Required for providers like GitHub Copilot that apply bad defaults.
		thinkingLevel: historian?.thinking_level,
		executeThresholdPercentage: config.execute_threshold_percentage,
		executeThresholdTokens: config.execute_threshold_tokens,
		commitClusterTrigger: config.commit_cluster_trigger,
		protectedTags: config.protected_tags,
		clearReasoningAge: config.clear_reasoning_age,
		historyBudgetPercentage: config.history_budget_percentage,
		memoryEnabled: config.memory.enabled,
		autoPromote: config.memory.auto_promote,
		userMemoriesEnabled: userMemoryCollectionEnabled(config.dreamer),
		language: config.language,
	};
}

function resolveAutoSearchFromConfig(
	config: MagicContextConfig,
): PiAutoSearchHandlerOptions {
	const auto = config.memory.auto_search;
	const enabled = auto?.enabled ?? false;
	return {
		enabled,
		scoreThreshold: auto?.score_threshold ?? 0.55,
		minPromptChars: auto?.min_prompt_chars ?? 20,
	};
}

export function resolveDreamerFromConfig(
	config: MagicContextConfig,
): DreamerConfig | undefined {
	return config.dreamer?.disable === true ? undefined : config.dreamer;
}

/**
 * Pi extension default export. Called once per Pi session.
 *
 * Registers the full Magic Context Pi runtime: tools, transform pipeline
 * (tagging + drops), historian trigger, nudges, auto-search hint,
 * /ctx-aug command, system-prompt injection, and dreamer scheduling.
 * All driven by the user's `magic-context.jsonc` (Pi convention paths).
 */
export default async function (pi: ExtensionAPI): Promise<void> {
	if (process.env[MAGIC_CONTEXT_PI_SUBAGENT_ENV] === "1") {
		log(
			`${PREFIX} subagent child detected (${MAGIC_CONTEXT_PI_SUBAGENT_ENV}=1); skipping full extension registration`,
		);
		return;
	}

	if (process.env.PI_SUBAGENT_CHILD === "1") {
		magicContextSubagentExtension(pi);
		return;
	}

	const storageDir = getMagicContextStorageDir();
	const dbPath = join(storageDir, "context.db");

	let db: ContextDatabase | null | undefined;
	try {
		db = openDatabase();
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		warn(
			`Magic Context (pi) failed to open SQLite store at ${dbPath}: ${message}. ` +
				"Plugin will not register hooks; storage path is unwritable or corrupt.",
		);
		return;
	}

	// openDatabase() FAILS CLOSED by returning null (not throwing) on the schema
	// fence — when the shared cross-harness DB has been migrated to a schema newer
	// than this binary supports (e.g. a newer OpenCode/Pi build ran a migration).
	// The try/catch above only catches genuine open exceptions, so without this
	// guard `db` is null and the very next call (runDeferredV22Backfill) would
	// crash the whole Pi process. Degrade gracefully instead: skip hook
	// registration until the binary is upgraded. The detailed reason is already
	// logged by openDatabase ("storage fatal: … newer than this binary supports").
	if (!db) {
		warn(
			`Magic Context (pi) storage unavailable at ${dbPath} (cache schema is newer than this binary supports). ` +
				"A pinned or stale plugin is likely sharing this database with a newer instance. " +
				"Plugin will not register hooks; run 'npx @cortexkit/magic-context@latest doctor --force' " +
				"(or update Pi/OpenCode) and restart to recover.",
		);
		return;
	}
	// Non-null const alias: `db` is a `let` (reassigned during open), so TS
	// won't carry the `!db` narrowing into the closures below. Capture it.
	const database: ContextDatabase = db;

	// v22 deferred legacy-memory identity backfill. openDatabase() has already
	// run migrations; the runner is fire-and-forget and logs failures without
	// blocking Pi startup.
	runDeferredV22Backfill(db).catch((err) => {
		warn(`[v22-backfill] background runner failed: ${err}`);
	});

	// Capture boot project for initial config load and logging only. Runtime
	// identity/path resolution uses ctx.cwd per hook/command so session cwd
	// switches follow the active project without reloading config.
	const projectDir = process.cwd();
	const projectIdentity = resolveProjectIdentity(projectDir);
	const seenDreamerProjectIdentities = new Set<string>([projectIdentity]);

	try {
		const pendingPiMarkerSessions = getSessionsWithPendingPiMarker(db);
		for (const sid of pendingPiMarkerSessions) {
			signalPiDeferredHistoryRefresh(sid);
			signalPiPendingMaterialization(sid);
		}
		if (pendingPiMarkerSessions.length > 0) {
			log(
				`${PREFIX} rehydrated ${pendingPiMarkerSessions.length} Pi deferred compaction marker session(s)`,
			);
		}
	} catch (err) {
		warn(
			`Magic Context (pi) failed to rehydrate deferred Pi compaction markers: ${err instanceof Error ? err.message : String(err)}`,
		);
	}

	info(
		`loaded v${PLUGIN_VERSION} | harness=pi | db=${dbPath} | ` +
			`project=${projectIdentity} | dir=${projectDir}`,
	);

	// Step 5b: load the user's full magic-context.jsonc config. The loader
	// reads $cwd/.pi/magic-context.jsonc and ~/.pi/agent/magic-context.jsonc
	// (Pi convention), validates them through the shared Zod schema, falls
	// back to defaults for invalid fields per-key, and returns merged
	// config + warnings.
	//
	// We surface warnings via the standard `warn()` channel so users see
	// them in the magic-context log. Loading never throws — bad config
	// gracefully degrades to defaults.
	ensureConfigLocationsMigrated(projectDir);
	const { config, warnings, loadedFromPaths } = loadPiConfig({
		cwd: projectDir,
	});
	if (loadedFromPaths.length > 0) {
		info(`config loaded from: ${loadedFromPaths.join(", ")}`);
	} else {
		info("config: no magic-context.jsonc found, using schema defaults");
	}
	for (const w of warnings) {
		warn(`config: ${w}`);
	}

	// Pi opens the shared DB before config is available (above), so apply the
	// configured SQLite tuning to the already-open connection now. cache_size /
	// mmap_size take effect live; future opens in this process pick them up via
	// setSqlitePragmaConfig.
	setSqlitePragmaConfig({
		cacheSizeMb: config.sqlite.cache_size_mb,
		mmapSizeMb: config.sqlite.mmap_size_mb,
	});
	applySqliteTuningPragmas(db);

	// Debug data-collection toggle: keep subagent child sessions instead of
	// deleting on success (parity with the OpenCode plugin).
	setKeepSubagents(config.keep_subagents === true);

	// Top-level disable: when `enabled: false` is set in config, register
	// nothing — same fail-closed posture the OpenCode plugin uses.
	if (!config.enabled) {
		info("plugin DISABLED via config (enabled: false) — skipping registration");
		return;
	}

	await ensureProjectRegisteredFromPiDirectory(projectDir, db);
	info(`registered embedding config for project ${projectIdentity}`);

	// Register the agent-facing tools. Reuses the same business logic
	// the OpenCode plugin uses (insertMemory, unifiedSearch, addNote, …)
	// via the shared cortexkit DB. Cross-harness memory sharing is automatic
	// because both plugins resolve the same project identity for the same
	// directory.
	registerMagicContextTools(pi, {
		db,
		ensureProjectRegistered: ensureProjectRegisteredFromPiDirectory,
		// Main extension entry never gets the dreamer-only ctx_memory
		// surface — those actions are reserved for dreamer subagents
		// loaded via subagent-entry.ts with the
		// `--magic-context-dreamer-actions` flag.
		allowDreamerActions: false,
		// ALWAYS register ctx_memory in the main entry. Pi is a single REPL that
		// can `/cd` between projects, but tool registration happens once at boot,
		// so gating registration on the BOOT project's memory.enabled would
		// mismatch the per-project prompt (which re-resolves memory.enabled each
		// pass): start in a memory-off project and switch to a memory-on one and
		// the tool would be absent while the prompt advertises it. The tool's
		// own per-call guard (ctx-memory.ts, getProjectEmbeddingSnapshot) refuses
		// when the CURRENT project has memory off, so always-register is correct.
		// (The subagent entry still uses memoryToolEnabled to keep ctx_memory off
		// the retrieval-only sidekick, a separate security concern.)
		memoryToolEnabled: true,
		// Match OpenCode's gating: when ctx_reduce_enabled is false,
		// we don't surface the tool at all (along with disabling §N§
		// prefix injection and stripping ctx_reduce mentions from the
		// system prompt). When true, register ctx_reduce.
		ctxReduceEnabled: config.ctx_reduce_enabled === true,
		protectedTags: config.protected_tags ?? 20,
		// Smart notes (surface_condition) only work when dreamer is
		// running — otherwise the note sits `pending` forever with no
		// path to surface. Match the user's dreamer config flag.
		dreamerEnabled: isDreamerRunnable(config),
	});
	info(
		`registered tools: ctx_search, ctx_memory, ctx_note, ctx_expand${
			config.ctx_reduce_enabled === true ? ", ctx_reduce" : ""
		}`,
	);

	// Register the per-LLM-call transform pipeline. Tags eligible message
	// parts via the shared Tagger and applies queued drops from
	// `pending_ops` so /ctx-flush and ctx_reduce work against Pi sessions.
	const historianConfig = resolveHistorianFromConfig(config);
	if (historianConfig) {
		historianConfig.onStatusChange = (ctx) => {
			updateStatusLine(ctx, {
				db,
				projectIdentity: resolveCurrentProject(ctx).projectIdentity,
			});
		};
	}
	const autoSearchConfig = resolveAutoSearchFromConfig(config);

	// Per-cwd context-handler options. Pi can switch projects mid-process
	// (`/cd`, multi-root); a switched-into checkout may carry its own
	// .pi/magic-context.jsonc (protected_tags, thresholds, memory/key-files
	// toggles, historian model). buildPiContextHandlerOptions resolves the
	// options for a given config; resolveContextOptionsForProject memoizes by
	// projectDir so the hot path (`pi.on("context")`, once per LLM call) does
	// one config read per distinct project and reuses it thereafter. The launch
	// cwd is pre-seeded with the already-loaded boot config.
	const contextOptionsByDir = new Map<string, PiContextHandlerOptions>();
	const buildContextOptions = (
		cfg: MagicContextConfig,
		hist: PiHistorianOptions | undefined,
		auto: PiAutoSearchHandlerOptions,
	): PiContextHandlerOptions => ({
		db: database,
		ctxReduceEnabled: cfg.ctx_reduce_enabled,
		smartDrops: cfg.smart_drops === true,
		protectedTags: cfg.protected_tags ?? 20,
		heuristics: {
			caveman:
				cfg.ctx_reduce_enabled === false && cfg.caveman_text_compression
					? {
							enabled: cfg.caveman_text_compression.enabled,
							minChars: cfg.caveman_text_compression.min_chars,
						}
					: undefined,
			clearReasoningAge: cfg.clear_reasoning_age,
		},
		injection: {
			memoryEnabled: cfg.memory.enabled,
			injectDocs: cfg.dreamer?.inject_docs !== false,
			injectionBudgetTokens: cfg.memory.injection_budget_tokens,
			temporalAwareness: cfg.temporal_awareness === true,
		},
		scheduler: {
			executeThresholdPercentage: cfg.execute_threshold_percentage,
			executeThresholdTokens: cfg.execute_threshold_tokens,
		},
		historian: hist,
		language: cfg.language,
		autoSearch: auto,
		resolveForProject: resolveContextOptionsForProject,
		maybeAutoEmbedSession: (sessionId, dir, identity) => {
			maybeAutoEmbedPiSession(
				{
					db: database,
					projectDir: dir,
					projectIdentity: identity,
					memoryEnabled: cfg.memory.enabled,
				},
				sessionId,
				dir,
				identity,
				(text) => {
					pi.sendMessage(
						{
							customType: "ctx-status",
							content: text,
							display: true,
						} as never,
						{ triggerTurn: false },
					);
				},
			);
		},
	});
	function resolveContextOptionsForProject(
		dir: string,
	): PiContextHandlerOptions {
		const cached = contextOptionsByDir.get(dir);
		if (cached) return cached;
		// A different checkout: re-resolve config + historian/auto-search from
		// the new cwd. The launch dir is pre-seeded below so this branch only
		// runs for genuine switches.
		ensureConfigLocationsMigrated(dir);
		const switchedConfig = loadPiConfig({ cwd: dir }).config;
		const switchedHistorian = resolveHistorianFromConfig(switchedConfig);
		if (switchedHistorian) {
			switchedHistorian.onStatusChange = (ctx) => {
				updateStatusLine(ctx, {
					db: database,
					projectIdentity: resolveCurrentProject(ctx).projectIdentity,
				});
			};
		}
		const built = buildContextOptions(
			switchedConfig,
			switchedHistorian,
			resolveAutoSearchFromConfig(switchedConfig),
		);
		contextOptionsByDir.set(dir, built);
		return built;
	}
	const bootContextOptions = buildContextOptions(
		config,
		historianConfig,
		autoSearchConfig,
	);
	contextOptionsByDir.set(projectDir, bootContextOptions);
	registerPiContextHandler(pi, bootContextOptions);
	info(
		historianConfig
			? `registered historian trigger (model=${historianConfig.model}, executeThreshold=${historianConfig.executeThresholdPercentage ?? 65}%)`
			: "registered historian trigger: DISABLED (set historian.model in magic-context.jsonc)",
	);
	info(
		autoSearchConfig.enabled
			? `registered auto-search hint (threshold=${autoSearchConfig.scoreThreshold}, minChars=${autoSearchConfig.minPromptChars})`
			: "registered auto-search hint: DISABLED (memory.auto_search.enabled=false)",
	);

	// Register the /ctx-aug slash command. Sidekick config is read straight
	// from `config.sidekick` — when disabled or missing a model, the command
	// surfaces a "not configured" message instead of attempting to run.
	const sidekickConfig: PiSidekickConfig | undefined =
		resolveSidekickFromConfig(config);
	registerCtxAugCommand(pi, sidekickConfig);
	info(
		sidekickConfig
			? `registered /ctx-aug (sidekick model=${sidekickConfig.model})`
			: "registered /ctx-aug (sidekick disabled — set sidekick.disable=false and sidekick.model in config)",
	);

	// Step 5c: register the four diagnostic/admin slash commands so Pi
	// reaches command-surface parity with the OpenCode plugin. All four
	// commands emit `pi.sendMessage(..., { triggerTurn: false })` — they
	// are never visible to the LLM and never trigger a turn. They mirror
	// the behavior of OpenCode's command-handler.ts but use Pi-native
	// surfaces (registerCommand + sendMessage) instead of OpenCode's
	// command.execute.before hook.
	registerCtxStatusCommand(pi, {
		db,
		projectIdentity,
		resolveProject: resolveCurrentProject,
		protectedTags: config.protected_tags,
		executeThresholdPercentage: config.execute_threshold_percentage,
		historyBudgetPercentage: config.history_budget_percentage,
		injectionBudgetTokens: config.memory?.injection_budget_tokens,
		commitClusterTrigger: config.commit_cluster_trigger,
		executeThresholdTokens: config.execute_threshold_tokens,
		dreamer: {
			runnable: isDreamerRunnable(config),
			scheduleSummary: summarizeDreamSchedule(config.dreamer),
		},
	});
	info("registered /ctx-status");
	registerStatusLine(pi, { db, projectIdentity });
	info("registered magic-context status line");

	registerCtxFlushCommand(pi, { db });
	info("registered /ctx-flush");

	// /ctx-recomp uses its own PiSubagentRunner instance — recomp can run
	// concurrently with normal historian, and giving each its own runner
	// avoids cross-cancellation. Same model + fallback chain as historian.
	registerCtxRecompCommand(pi, {
		db,
		runner: new PiSubagentRunner(),
		historianModel: historianConfig?.model,
		historianChunkTokens: deriveHistorianChunkTokens(
			resolveHistorianContextLimit(historianConfig?.model),
		),
		historianFallbacks: historianConfig?.fallbackModels,
		historianTimeoutMs: config.historian_timeout_ms,
		historianThinkingLevel: historianConfig?.thinkingLevel,
		language: config.language,
		memoryEnabled: config.memory.enabled,
		autoPromote: config.memory.auto_promote,
	});
	info("registered /ctx-recomp");

	// E6b/E6c: /ctx-session-upgrade — full recomp (legacy→v2 tiered) + once-per-
	// project memory migration into the 5-category taxonomy. Own runner instance
	// for the same isolation reasons as /ctx-recomp.
	registerCtxSessionUpgradeCommand(pi, {
		db,
		runner: new PiSubagentRunner(),
		historianModel: historianConfig?.model,
		historianChunkTokens: deriveHistorianChunkTokens(
			resolveHistorianContextLimit(historianConfig?.model),
		),
		historianFallbacks: historianConfig?.fallbackModels,
		historianTimeoutMs: config.historian_timeout_ms,
		historianThinkingLevel: historianConfig?.thinkingLevel,
		language: config.language,
		memoryEnabled: config.memory.enabled,
		autoPromote: config.memory.auto_promote,
		userMemoriesEnabled: userMemoryCollectionEnabled(config.dreamer),
	});
	info("registered /ctx-session-upgrade");

	registerCtxDreamCommand(pi, {
		db,
		projectDir,
		projectIdentity,
		resolveProject: resolveCurrentProject,
		dreamerEnabled: isDreamerRunnable(config),
		onProjectSeen: (identity) => seenDreamerProjectIdentities.add(identity),
	});
	info("registered /ctx-dream");

	registerCtxEmbedCommand(pi, {
		db,
		projectDir,
		projectIdentity,
		memoryEnabled: config.memory.enabled,
		resolveProject: resolveCurrentProject,
	});
	info("registered /ctx-embed");

	// Register Pi project with the singleton dreamer timer. When dreamer is
	// disabled in config (default) this is a no-op. When enabled, the timer
	// schedules dream runs based on config.dreamer.schedule and uses
	// PiSubagentRunner to spawn child sessions for each task.
	const dreamerConfig = resolveDreamerFromConfig(config);
	if (dreamerConfig) {
		registerPiDreamerProject({
			db,
			projectDir,
			projectIdentity,
			config: dreamerConfig,
			// Council finding #7: thread real embedding + memory config so
			// dreamer can do semantic dedup AND can write memory updates.
			// Previously hardcoded to off/false, making most dreamer tasks
			// useless on Pi.
			embeddingConfig: config.embedding,
			memoryEnabled: config.memory.enabled,
			language: config.language,
			gitCommitIndexing: config.memory.git_commit_indexing,
			onAdjunctsRefreshNeeded: signalPiSystemPromptRefreshForProject,
		});
		info(`registered dreamer (${summarizeDreamSchedule(dreamerConfig)})`);
	} else {
		info(
			isDreamerRunnable(config)
				? "registered dreamer: DISABLED (no dreamer config)"
				: "registered dreamer: DISABLED (dreamer.disable=true or no dreamer config)",
		);
	}

	// Inject the magic-context guidance block into the system prompt for every agent
	// turn, then run hash-detection + sticky-date freezing so the
	// resulting prompt stays cache-stable across turns when nothing
	// material has changed.
	//
	// Pi has prefix caching the same way OpenCode does — every major
	// LLM provider (Anthropic, OpenAI, Codex, GitHub Copilot, etc.)
	// caches the system prompt portion of the prefix. Drift between
	// turns busts the cache and the user pays full input price for the
	// next call. The protections here mirror OpenCode's
	// `experimental.chat.system.transform` handler in
	// `system-prompt-hash.ts`.
	pi.on("before_agent_start", async (event, ctx) => {
		// Startup release announcement (Pi parity with OpenCode TUI dialog +
		// Desktop ignored message). Fires once per ANNOUNCEMENT_VERSION across
		// the whole machine — persistence file is shared with the OpenCode
		// plugin via `getMagicContextStorageDir()/last_announced_version`.
		//
		// Skipped silently when:
		//   - announcement constants are empty (bugfix-only release)
		//   - the current ANNOUNCEMENT_VERSION was already dismissed (here or
		//     in OpenCode TUI/Desktop)
		//   - ctx.hasUI is false (print/rpc subagent — no point notifying)
		//
		// Fire-and-forget: storage write happens inside markAnnouncementSeen,
		// any failure is swallowed. Worst case is a duplicate notification
		// the next time the user starts an interactive Pi session.
		try {
			if (ctx.hasUI && shouldShowAnnouncement()) {
				// URLs render as plain text. Modern terminals auto-detect and
				// let users Cmd-click; older terminals require manual copy.
				// We previously wrapped URLs in OSC 8 hyperlink escapes, but
				// not all terminals support them and `ctx.ui.notify` may also
				// re-render the message through pi-tui's text pipeline that
				// strips raw escapes. Plain text is the most reliable surface.
				const featureText = ANNOUNCEMENT_FEATURES.map(
					(line) => `  • ${line}`,
				).join("\n");
				const sections = [
					`✨ Magic Context v${ANNOUNCEMENT_VERSION} — what's new:`,
					"",
					featureText,
				];
				if (ANNOUNCEMENT_FOOTER && ANNOUNCEMENT_FOOTER.trim().length > 0) {
					// Blank-line separator distinguishes the persistent footer
					// (Discord invite, etc.) from the version-specific bullets.
					sections.push("", ANNOUNCEMENT_FOOTER);
				}
				ctx.ui.notify(sections.join("\n"), "info");
				markAnnouncementSeen(ANNOUNCEMENT_VERSION);
			}
		} catch {
			// Never block agent start on announcement delivery.
		}

		try {
			const currentProject = resolveCurrentProject(ctx);
			seenDreamerProjectIdentities.add(currentProject.projectIdentity);

			// Re-register the dreamer for the CURRENT project. The boot-time
			// registration above used process.cwd(), but Pi can switch projects
			// mid-process (`/cd`, multi-root). Without this, a switched-into
			// project is never dreamed and `/ctx-dream` there throws
			// "not registered". registerPiDreamerProject is idempotent for the
			// same identity+dir, and rebuilds against the new checkout when the
			// directory changed (worktree/clone of the same repo).
			//
			// On a genuine switch we MUST re-resolve config from the new
			// checkout's cwd — the boot `config`/`dreamerConfig` belong to the
			// launch directory, and a switched-into project may carry its own
			// .pi/magic-context.jsonc (different model/schedule, or its own
			// `dreamer.disable`). Reusing the boot config would silently run the
			// dreamer in the new checkout with the old project's settings.
			const switchedProject = currentProject.projectDir !== projectDir;
			if (switchedProject)
				ensureConfigLocationsMigrated(currentProject.projectDir);
			const effectiveConfig = switchedProject
				? loadPiConfig({ cwd: currentProject.projectDir }).config
				: config;
			const effectiveDreamerConfig = switchedProject
				? resolveDreamerFromConfig(effectiveConfig)
				: dreamerConfig;
			if (effectiveDreamerConfig) {
				try {
					registerPiDreamerProject({
						db,
						projectDir: currentProject.projectDir,
						projectIdentity: currentProject.projectIdentity,
						config: effectiveDreamerConfig,
						embeddingConfig: effectiveConfig.embedding,
						memoryEnabled: effectiveConfig.memory.enabled,
						gitCommitIndexing: effectiveConfig.memory.git_commit_indexing,
						onAdjunctsRefreshNeeded: signalPiSystemPromptRefreshForProject,
					});
				} catch (err) {
					warn("before_agent_start: registerPiDreamerProject threw:", err);
				}
			} else if (switchedProject) {
				// The switched-into checkout DISABLES the dreamer. Any existing
				// registration for this identity is pinned to the old checkout and
				// still enabled — registerPiDreamerProject's disable early-return
				// can't clean it up, so tear it down explicitly here.
				try {
					unregisterPiDreamerProject({
						projectIdentity: currentProject.projectIdentity,
					});
				} catch (err) {
					warn("before_agent_start: unregisterPiDreamerProject threw:", err);
				}
			}
			// Pi exposes `sessionManager.getSessionId()` once a session is
			// active. We resolve it here defensively because before_agent_start
			// fires once per agent turn.
			const sm = ctx.sessionManager;
			let sessionId: string | undefined;
			if (sm !== undefined) {
				const getId = (sm as { getSessionId?: () => string | undefined })
					.getSessionId;
				if (typeof getId === "function") {
					try {
						const id = getId.call(sm);
						if (typeof id === "string" && id.length > 0) sessionId = id;
					} catch {
						// Fail open — sessionId stays undefined.
					}
				}
			}
			if (sessionId) {
				trackSessionForProject(currentProject.projectIdentity, sessionId);

				// Re-arm a pending Pi compaction-marker drain on session ACTIVATION,
				// not just at process startup. session_before_switch clears the
				// in-memory deferred-refresh/materialization sets for the outgoing
				// session (those Sets are per-process and would otherwise leak); but
				// the durable pending marker in session_meta survives. On switch-BACK
				// the marker would then sit undrained (the drain is signal-driven, and
				// startup-only rehydration never re-fires). Re-signal here when this
				// session has a durable pending marker so the next pass drains it.
				//
				// Gate on the same APIs the drain itself requires
				// (sessionManager.appendCompaction + getBranch): when they're
				// unavailable the drain skips-and-PRESERVES the signal, so
				// re-signaling every turn would force materialization repeatedly
				// with no way to make progress (a per-turn cache bust). Only re-arm
				// when the marker can actually be applied.
				try {
					const smForDrain = sm as {
						appendCompaction?: unknown;
						getBranch?: unknown;
					};
					const canDrain =
						typeof smForDrain.appendCompaction === "function" &&
						typeof smForDrain.getBranch === "function";
					if (canDrain && getPendingPiCompactionMarkerState(db, sessionId)) {
						signalPiDeferredHistoryRefresh(sessionId);
						signalPiPendingMaterialization(sessionId);
					}
				} catch {
					// Best-effort: a read failure must not block agent start.
				}

				// E6d: one-time upgrade reminder for sessions with legacy (pre-v2)
				// compartments. Model-invisible (ctx.ui.notify), self-gating via the
				// durable + per-process guards in the shared helper. Only when the
				// historian can run (so /ctx-session-upgrade is actionable).
				if (ctx.hasUI && historianConfig?.model) {
					void maybeSendUpgradeReminder(
						{
							client: null,
							db,
							sendIgnoredMessage: async (_client, _sid, text) => {
								ctx.ui.notify(text, "info");
								return "sent";
							},
							getNotificationParams: () => ({}),
							// Pi's ctx.ui.notify is a TRANSIENT toast (no scrollback),
							// so the durable stamp must not suppress after one missed
							// toast — re-prompt each Pi start until the session upgrades.
							deliveryPersists: false,
						},
						sessionId,
					).catch(() => {
						// Never block agent start on reminder delivery.
					});
				}
			}

			// Use effectiveConfig (re-resolved from the CURRENT checkout's cwd on
			// a project switch) for every system-prompt decision below — a
			// switched-into project may carry its own .pi/magic-context.jsonc
			// (memory/docs/key-files/injection toggles). Reusing boot `config`
			// would render the launch project's adjuncts in the new checkout.
			if (effectiveConfig.system_prompt_injection?.enabled === false) {
				return;
			}
			const skipSigs =
				effectiveConfig.system_prompt_injection?.skip_signatures ?? [];
			if (
				skipSigs.some(
					(sig) => sig.length > 0 && event.systemPrompt.includes(sig),
				)
			) {
				return;
			}

			// PEEK the system-prompt refresh signal. Set by:
			//   - `/ctx-flush`
			//   - dreamer publication of new ARCHITECTURE.md / STRUCTURE.md
			//   - user-memory promotion (dreamer)
			//   - hash-change detection on the previous turn (signaled below)
			//
			// When set, we re-read disk-backed adjuncts on this turn. When
			// not set, cached values are reused.
			//
			// PEEK-then-drain-on-success (Oracle audit Round 8 #6): we
			// only `clearSystemPromptRefresh(...)` AFTER the rebuild
			// (`buildMagicContextBlock` + `processSystemPromptForCache`)
			// completes successfully. If either throws, the flag survives
			// so the next prompt retries the rebuild.
			const isCacheBusting = sessionId
				? hasSystemPromptRefresh(sessionId)
				: true; // first-pass-no-session: act as cache-busting (force fresh read)

			const effectiveDreamerRunnable = switchedProject
				? isDreamerRunnable(effectiveConfig)
				: isDreamerRunnable(config);
			const block = buildMagicContextBlock({
				db,
				cwd: currentProject.projectDir,
				sessionId,
				memoryEnabled: effectiveConfig.memory.enabled,
				includeGuidance: true,
				protectedTags: effectiveConfig.protected_tags,
				ctxReduceEnabled: effectiveConfig.ctx_reduce_enabled,
				dreamerEnabled: effectiveDreamerRunnable,
				temporalAwarenessEnabled: effectiveConfig.temporal_awareness ?? false,
				cavemanTextCompressionEnabled:
					effectiveConfig.ctx_reduce_enabled === false &&
					effectiveConfig.caveman_text_compression?.enabled === true,
				language: effectiveConfig.language,
				// Stable user memories rendered as <user-profile> — dreamer
				// promotes recurring observations into this set, then the
				// system prompt surfaces them across all sessions in the
				// project. Gated on dreamer.user_memories.enabled.
				userMemoriesEnabled: userMemoryCollectionEnabled(
					effectiveConfig.dreamer,
				),
				isCacheBusting,
				existingSystemPrompt: event.systemPrompt,
			});

			// Compose the final system prompt: base prompt from Pi + our
			// magic-context block. We always run hash detection on the
			// composed string so even sessions with no data block (e.g.
			// memories disabled, no docs, no key files) still get
			// sticky-date freezing and hash-change tracking.
			const composedPrompt = block
				? `${event.systemPrompt}\n\n${block}`
				: event.systemPrompt;

			if (!sessionId) {
				// No session id yet — return the composed prompt without
				// cache logic. The next turn (with a session id) will
				// compute the first hash and set sticky date.
				if (block) return { systemPrompt: composedPrompt };
				return;
			}

			const result = processSystemPromptForCache({
				db,
				sessionId,
				systemPrompt: composedPrompt,
				isCacheBusting,
			});

			if (result.hashChanged) {
				// Real prompt-content change. Cache prefix is already
				// busted on this turn. Signal all three independent
				// refresh sets so the next pi.on("context") event
				// rebuilds <session-history> + lets queued ops
				// materialize, AND the next before_agent_start refreshes
				// adjuncts (since this turn's adjunct read used the
				// cached values that are now potentially stale).
				signalPiHistoryRefresh(sessionId);
				signalPiSystemPromptRefresh(sessionId);
				signalPiPendingMaterialization(sessionId);
			}

			// PEEK-then-drain-on-success (Oracle audit Round 8 #6):
			// drain only if the start-of-pass peek was true. Using the
			// CAPTURED boolean (not a re-read of the set) so that a
			// signal added later in the same pass — e.g. `result.hashChanged`
			// just above — survives to the next prompt for retry.
			if (isCacheBusting) {
				clearSystemPromptRefresh(sessionId);
			}

			return { systemPrompt: result.systemPrompt };
		} catch (error) {
			warn("failed to build magic-context block:", error);
			return;
		}
	});
	info("registered before_agent_start system prompt injector");

	// agent_end MUST be fire-and-forget for in-flight historian / dreamer
	// runs.
	//
	// REGRESSION FIXED HERE: Earlier code awaited `awaitInFlightHistorians()`
	// inside this handler with the (incorrect) assumption that Pi's event
	// fanout is synchronous and ignores returned Promises. In reality
	// pi-coding-agent's `extensions/runner.js` does `await handler(event, ctx)`
	// for every extension `agent_end` handler, and `agent-session.js`
	// awaits its own emit before delivering the UI-facing `agent_end`.
	// The TUI loader stops only after that UI event. Net effect: every
	// turn that triggered a historian (a 30s+ background subagent) left
	// the user staring at "Working..." with `historian` pinned in the
	// footer until the background run finished — the OPPOSITE of the
	// "compact in the background while the main agent keeps working"
	// invariant magic-context is supposed to provide.
	//
	// Why fire-and-forget is safe in interactive mode:
	//   - The Pi process stays alive between turns. The next turn's
	//     `pi.on("context")` handler checks `inFlightHistorian.has(sessionId)`
	//     and skips re-firing while a previous historian is still
	//     running, so we never double-spawn.
	//   - Historian publication paths register the run promise in
	//     `inFlightHistorian` so emergency 95% waits and `session_shutdown`
	//     drainage can still join the background work when actually
	//     needed.
	//   - All work historian does is durable (compartment + fact rows,
	//     publish marker, signalPiHistoryRefresh). Even if the user
	//     closes Pi mid-historian and the subprocess gets killed, the
	//     next session start re-evaluates and either picks up where the
	//     prior run left off or recovers from `historian_failure_count`.
	//
	// `pi --print` (single-turn, exits after agent_end) is the one mode
	// where backgrounding is genuinely incompatible with subprocess
	// lifetime — Pi's process exits and SIGKILLs the still-running
	// historian. That tradeoff is intentional: print mode is for
	// scripting / one-shot tasks where blocking the user's interactive
	// shell on a 30s historian is also wrong, just in a different way.
	// We let print mode skip the wait too. Users who want guaranteed
	// historian completion in print mode should run interactive Pi
	// instead.
	pi.on("agent_end", (event, ctx) => {
		// Synchronous return — DO NOT await background work here.
		// awaitInFlightHistorians()/awaitInFlightDreamers() are still
		// invoked at session_shutdown where they belong (and where pi
		// gives us a window before tearing down stdio). Errors from
		// background runs are handled by their own try/catch chains
		// (runPiHistorian wraps everything; spawnPiHistorianRun's
		// .finally cleans up the inFlight map).
		log("agent_end: returning synchronously (background work continues)");

		// Channel 2 (ceiling) nudge delivery — the Pi analog of OpenCode's
		// event-handler delivery on terminal message.updated. The pipeline
		// records a `pending` intent near the threshold; deliver it here at the
		// turn boundary via sendUserMessage(followUp). Internally CAS-gated to
		// one delivery per session lifetime, and no-ops unless `pending`.
		// Fire-and-forget; never block agent_end.
		//
		// Deliver ONLY on a clean final stop. Pi emits agent_end for error /
		// aborted responses and for retry attempts too (agent-loop); delivering
		// on those would inject the follow-up mid-retry and burn the one-shot cap
		// before the turn actually completed. OpenCode's equivalent gates on
		// finish === "stop". Mirror that with the final assistant's stopReason.
		try {
			const msgs = (
				event as { messages?: Array<{ role?: string; stopReason?: string }> }
			)?.messages;
			const lastAssistant = Array.isArray(msgs)
				? [...msgs].reverse().find((m) => m?.role === "assistant")
				: undefined;
			if (lastAssistant?.stopReason === "stop") {
				const sessionId = ctx.sessionManager?.getSessionId?.();
				if (sessionId && db) maybeDeliverChannel2Pi(pi, db, sessionId);
			}
		} catch (err) {
			log(`agent_end: channel2 delivery skipped: ${String(err)}`);
		}
	});

	// Tool-execution-start hook: detect note-nudge triggers from
	// agent tool usage. Mirrors OpenCode's `tool.execute.after` hook in
	// `hook-handlers.ts` (`createToolExecuteAfterHook`). We use Pi's
	// `tool_execution_start` event because (a) it fires before the tool
	// runs (so we can inspect args without waiting for output, matching
	// OpenCode's `tool.execute.before`/`after` that have full args
	// available), and (b) `tool_execution_end` is fire-and-forget and
	// could race with the next pipeline pass.
	//
	// What we wire:
	//
	//   - `todowrite` with all-terminal todos → `todos_complete` trigger.
	//     The agent's `todos` arg is an array of {id, content, status}
	//     items. Note nudges should fire only when EVERY item is in a
	//     terminal state (`completed` or `cancelled`) — firing on every
	//     todowrite is too eager since agents call it repeatedly during
	//     work to mark intermediate progress.
	//
	//   - `ctx_note` (any action) → `clearNoteNudgeState(sessionId)`.
	//     The agent already saw / acted on notes, so we kill any
	//     pending sticky reminder for this session right away. Subagents
	//     never deliver note nudges (gated upstream in postprocess),
	//     so we still skip the trigger for them. Mirrors OpenCode's
	//     `if (typedInput.tool === "ctx_note") clearNoteNudgeState(...)`.
	pi.on("tool_execution_start", async (event, ctx) => {
		try {
			const sessionId = ctx.sessionManager.getSessionId();
			if (event.toolName === "todowrite") {
				const todoArgs = event.args as
					| { todos?: Array<{ status?: string }> }
					| undefined;
				const todos = todoArgs?.todos;
				const sessionMeta = Array.isArray(todos)
					? getOrCreateSessionMeta(db, sessionId)
					: null;

				// Synthetic-todowrite snapshot capture (Pi parity with
				// OpenCode hook-handlers.ts:386-401). Persist normalized
				// state on EVERY todowrite call so the transform-time
				// injection path in pi-pipeline.ts always has a current
				// snapshot to replay on the next cache-busting pass.
				// Cache-safe: this is a pure DB write with no message
				// mutation. Subagents skip — they do not get synthetic
				// todowrite injection.
				if (sessionMeta && !sessionMeta.isSubagent) {
					const normalizedTodos = normalizeTodoStateJson(todos);
					if (normalizedTodos !== null) {
						updateSessionMeta(db, sessionId, {
							lastTodoState: normalizedTodos,
						});
					}
				}

				if (
					Array.isArray(todos) &&
					todos.length > 0 &&
					todos.every(
						(t) => t.status === "completed" || t.status === "cancelled",
					)
				) {
					if (sessionMeta && !sessionMeta.isSubagent) {
						onNoteTrigger(db, sessionId, "todos_complete");
					}
				}
			} else if (event.toolName === "ctx_note") {
				clearNoteNudgeTriggerAndCooldown(db, sessionId);
			}
		} catch (err) {
			// tool-event hook is opportunistic; failure should not break
			// the agent loop.
			log(
				`tool_execution_start hook failed (continuing): ${err instanceof Error ? err.message : String(err)}`,
			);
		}
	});

	pi.on("tool_execution_end", async (event, ctx) => {
		try {
			const sessionId = ctx.sessionManager.getSessionId();
			if (typeof sessionId !== "string" || sessionId.length === 0) return;
			if (event.toolName === "ctx_reduce") {
				markPiChannel1Reduced(sessionId, db);
			}
		} catch (err) {
			log(
				`tool_execution_end hook failed (continuing): ${err instanceof Error ? err.message : String(err)}`,
			);
		}
	});

	// Channel 1 (ctx_reduce in-turn nudge), Pi parity with OpenCode's
	// `tool.execute.after` → `output.output` append. `tool_result` lets an
	// extension REPLACE the recorded tool result content; returning the original
	// content plus an appended `<system-reminder>` block persists to the session
	// JSONL (via `appendMessage` on `message_end`) and replays verbatim on every
	// later `context` pass — "free sticky", no anchor/CAS/replay machinery. The
	// metric baseline is computed in the pipeline (`pi.on("context")`) and read
	// here, exactly mirroring OpenCode's transform→tool.execute.after split.
	pi.on("tool_result", async (event, ctx) => {
		try {
			const sessionId = ctx.sessionManager.getSessionId();
			if (typeof sessionId !== "string" || sessionId.length === 0) return;
			// Channel 2 mid-turn delivery: a pending ceiling intent steers a
			// queued user message into the NEXT STEP of the in-flight turn so
			// the agent is warned while the pile is still growing (agent_end
			// stays as the idle fallback). No-ops unless pending + revalidated.
			if (db) maybeDeliverChannel2Pi(pi, db, sessionId, "steer");
			const block = maybeChannel1ReminderForToolResult({
				db,
				sessionId,
				toolName: event.toolName,
				content: event.content,
			});
			if (!block) return;
			return { content: [...event.content, block] };
		} catch (err) {
			log(
				`tool_result hook failed (continuing): ${err instanceof Error ? err.message : String(err)}`,
			);
		}
	});

	// Cancel Pi's native context compaction. Magic Context owns the
	// compacted view of conversation history through its own historian
	// pipeline (compartments + facts + memories rendered as
	// `<session-history>` in `pi.on("context")`). If Pi's auto-compaction
	// were to run, it would:
	//   1. Pack the full conversation into a single plain-text user
	//      message and ask the LLM to summarize it. For sessions with a
	//      lot of accumulated history (especially after `doctor migrate`)
	//      that summarization request itself overflows the model's
	//      context window — the failure mode that surfaced as
	//      "Context overflow recovery failed" on migrated sessions.
	//   2. Replace history with that flat summary and lose the structured
	//      compartment / fact / memory state we depend on.
	//
	// Returning `{ cancel: true }` aborts both the threshold-driven
	// auto-compact and the post-overflow recovery compact. Pi treats
	// the abort as a no-op and proceeds with the unmodified branch on
	// the next turn — at which point our `pi.on("context")` transform
	// shrinks the prompt via tag drops, caveman compression, and
	// `<session-history>` injection over the much smaller live tail.
	//
	// Steady-state sessions normally don't hit this path because our
	// historian writes a Pi compaction marker at the boundary
	// (`sessionManager.appendCompaction()`), so `getBranch()` already
	// trims the prefix before Pi ever evaluates `shouldCompact`. The
	// hook is the safety net for everything else: migrated sessions
	// without a compaction marker, sessions where historian failed,
	// or any future flow where Pi's heuristic decides to compact.
	pi.on("session_before_compact", async (_event, ctx) => {
		try {
			const sessionId = ctx.sessionManager?.getSessionId?.();
			if (typeof sessionId === "string" && sessionId.length > 0) {
				clearPiM0Cache(db, sessionId, "session_before_compact");
			}
		} catch {
			// best-effort; still cancel Pi native compaction below
		}
		info("session_before_compact: cancelling — magic-context owns compaction");
		return { cancel: true };
	});

	// Strip injected `§N§` tag prefix from assistant text BEFORE Pi
	// persists the message to disk and renders it to the UI. Mirrors
	// OpenCode's `experimental.text.complete` handler which scrubs the
	// prefix from `output.text` before the assistant message lands in
	// `opencode.db`.
	//
	// Pi's `agent-session.ts` emits `message_end` to extensions BEFORE
	// calling `sessionManager.appendMessage(event.message)`. Mutating
	// the message reference in this handler is therefore visible to
	// the persistence call — same effect as OpenCode's hook on a
	// different harness.
	//
	// Why this matters: LLMs frequently mimic the `§N§` prefix they
	// see on prior assistant messages and emit `§4§ Yes...` at the
	// start of a fresh response. The mimicry is harmless for cache
	// (we re-strip and re-inject on the next transform pass), but the
	// stored text is what Pi's UI renders — without this scrub, users
	// see internal tag IDs at the start of every assistant turn.
	pi.on("message_end", async (event, ctx) => {
		try {
			const msg = event.message as unknown;
			if (msg !== null && typeof msg === "object") {
				stripTagPrefixFromAssistantMessage(
					msg as { role: string; content: unknown },
				);
			}
		} catch (err) {
			warn("message_end: stripTagPrefixFromAssistantMessage threw:", err);
		}

		// Update last_response_time + last_input_tokens + last_context_percentage
		// so the scheduler's TTL gating can decide between execute and defer
		// on the next transform pass. Without this, every Pi pass would either
		// always execute (stale lastResponseTime=0 → TTL elapsed) or always
		// defer (no usage data) — neither matches OpenCode parity.
		try {
			const sm = ctx.sessionManager as
				| { getSessionId?: () => string | undefined }
				| undefined;
			const sessionId = sm?.getSessionId?.();
			if (typeof sessionId !== "string" || sessionId.length === 0) return;
			const endedMsg = event.message as unknown as {
				id?: string;
				role?: string;
			};
			if (
				endedMsg?.role === "assistant" &&
				typeof endedMsg.id === "string" &&
				endedMsg.id.length > 0
			) {
				const messageId = endedMsg.id;
				scheduleIncrementalIndex(db, sessionId, messageId, () => {
					const rawMessages = readPiSessionMessages(ctx);
					return (
						rawMessages.find((message) => message.id === messageId) ?? null
					);
				});
			}
			persistPiMessageEndModelMeta({
				db,
				sessionId,
				message: event.message,
				cacheTtlConfig: config.cache_ttl,
			});
			// Compute pressure with OpenCode-equivalent semantics: pull
			// the assistant's `usage` field and use
			// `input + cacheRead + cacheWrite` (NOT output) divided by
			// the effective context limit. The window comes from Pi's own
			// runtime — `getContextUsage().contextWindow`, falling back to
			// `ctx.model.contextWindow` if usage hasn't populated — NOT
			// models.dev. `session_meta.detected_context_limit` still overrides
			// it (in persistPiPressureFromMessageEnd) so post-overflow pressure
			// reflects the real, lower limit. See `pi-pressure.ts` for rationale.
			const piUsage = ctx.getContextUsage?.();
			const piContextWindow =
				piUsage &&
				typeof piUsage.contextWindow === "number" &&
				piUsage.contextWindow > 0
					? piUsage.contextWindow
					: (ctx.model?.contextWindow ?? 0);
			await persistPiPressureFromMessageEnd({
				db,
				sessionId,
				message: event.message,
				piContextWindow,
				piTokens:
					piUsage && typeof piUsage.tokens === "number"
						? piUsage.tokens
						: undefined,
				notifyIssue: async (message) => {
					const uiNotify = (
						ctx as { ui?: { notify?: (message: string) => unknown } }
					).ui?.notify;
					if (typeof uiNotify === "function") {
						void uiNotify.call(ctx.ui, message);
					} else {
						warn(message);
					}
				},
			});

			// Synthetic-todowrite capture (Pi parity with OpenCode
			// hook-handlers.ts `tool.execute.after` for `todowrite`).
			//
			// Why message_end and not tool_execution_start:
			//   Pi's `tool_execution_start` only fires for tools Pi has
			//   actually executed (i.e. tools the agent registered).
			//   The mocked todowrite in tests — and any user-driven
			//   custom todowrite-shaped tool that isn't in Pi's registry
			//   — would not trigger `tool_execution_start`. Reading the
			//   assistant message at `message_end` catches every
			//   todowrite-shaped `toolCall` block regardless of whether
			//   Pi could execute it locally, matching what OpenCode
			//   captures via `tool.execute.after` on every visible tool
			//   call.
			//
			// Cache safety: pure DB write, no message mutation.
			// Subagents skip — they don't get synthetic todowrite
			// injection downstream (mirrors OpenCode `fullFeatureMode`
			// gate).
			try {
				const sessionMetaForTodo = getOrCreateSessionMeta(db, sessionId);
				if (!sessionMetaForTodo.isSubagent) {
					const msg = event.message as
						| { role?: string; content?: unknown }
						| undefined;
					if (msg && msg.role === "assistant" && Array.isArray(msg.content)) {
						for (const block of msg.content) {
							if (!block || typeof block !== "object") continue;
							const b = block as {
								type?: unknown;
								name?: unknown;
								arguments?: unknown;
							};
							if (b.type !== "toolCall") continue;
							if (typeof b.name !== "string") continue;
							if (b.name !== "todowrite") {
								continue;
							}
							const args = b.arguments as
								| { todos?: unknown }
								| null
								| undefined;
							const todos = args?.todos;
							if (!Array.isArray(todos)) continue;
							const normalized = normalizeTodoStateJson(todos);
							if (normalized === null) continue;
							updateSessionMeta(db, sessionId, {
								lastTodoState: normalized,
							});
							// First valid todowrite block wins — mirrors OpenCode's
							// `tool.execute.after` behavior of capturing one
							// snapshot per tool invocation.
							break;
						}
					}
				}
			} catch (err) {
				warn("message_end: synthetic todowrite capture failed:", err);
			}
		} catch (err) {
			warn("message_end: persist session_meta usage failed:", err);
		}

		// Overflow recovery: if Pi's assistant message ended with a
		// provider context-overflow error (`message.errorMessage` matches
		// a known overflow pattern), record the recovery flag in
		// session_meta so the next transform pass treats this session as
		// "needs emergency recovery" — historian fires immediately, drop-
		// all-tools applies, and pressure math uses the real
		// detected_context_limit if the error reported one.
		//
		// Pi populates `errorMessage` on the assistant message when the
		// underlying API call fails (we saw exactly this pattern in the
		// Codex `context_length_exceeded` failure that motivated this
		// work). The provider-agnostic `detectOverflow` helper from
		// shared core matches Anthropic, OpenAI, Codex/OpenAI, xAI,
		// Cerebras, GitHub Copilot, OpenRouter, Ollama, vLLM, Mistral,
		// MiniMax, Kimi, Gemini, and a generic fallback.
		try {
			const sm = ctx.sessionManager as
				| { getSessionId?: () => string | undefined }
				| undefined;
			const sessionId = sm?.getSessionId?.();
			if (typeof sessionId !== "string" || sessionId.length === 0) return;
			const msgRaw = event.message as unknown;
			if (!msgRaw || typeof msgRaw !== "object") return;
			const msg = msgRaw as {
				role?: string;
				errorMessage?: string;
				provider?: string;
				model?: string;
			};
			if (msg.role !== "assistant") return;
			if (
				typeof msg.errorMessage !== "string" ||
				msg.errorMessage.length === 0
			) {
				return;
			}
			const detection = detectOverflow(msg.errorMessage);
			if (!detection.isOverflow) return;
			const modelKey =
				typeof msg.provider === "string" &&
				typeof msg.model === "string" &&
				msg.provider.length > 0 &&
				msg.model.length > 0
					? `${msg.provider}/${msg.model}`
					: undefined;
			recordOverflowDetected(db, sessionId, detection.reportedLimit, modelKey);
			log(
				`[magic-context][${sessionId}] overflow detected: reportedLimit=${
					detection.reportedLimit ?? "?"
				} pattern=${detection.matchedPattern ?? "?"}`,
			);
		} catch (err) {
			warn("message_end: overflow detection failed:", err);
		}
	});

	// Unregister project from dreamer timer on session shutdown. Pi's
	// `/reload` command tears down extensions and re-runs this default
	// export — without unregistering, the dreamer timer would hold a
	// stale reference to the previous extension instance.
	//
	// IMPORTANT: We do NOT close the SQLite handle here. `openDatabase()`
	// caches handles in a process-lifetime Map keyed by path; closing
	// the handle invalidates the cache entry, but the Map still returns
	// the closed handle on the next `openDatabase()` call after reload,
	// causing every tool/hook to fail with "database is not open". The
	// DB handle is intentionally process-lifetime — Pi's `/reload`
	// re-runs the extension code but keeps the host process alive, so
	// the cached handle is still valid across reload boundaries.
	pi.on("session_shutdown", async (_event, ctx) => {
		// Bounded drain of in-flight historian / dreamer runs that were
		// kicked off by recent turns. We moved the drain here from
		// `agent_end` because Pi awaits agent_end handlers and was
		// stalling the UI loader on every turn that triggered historian.
		// session_shutdown only fires when the user is actually leaving
		// the session, so a brief wait is acceptable — and lets the
		// JSONL session state reach a consistent compartment boundary
		// before the process exits.
		//
		// 5-second cap protects interactive shutdown from a hung
		// subagent. In `pi --print` mode the process exits after
		// agent_end before this handler fires anyway, so the cap
		// doesn't help that mode (and we don't pretend it does — see
		// the comment block on the agent_end handler above).
		const SHUTDOWN_DRAIN_MS = 5_000;
		try {
			await withTimeout(awaitInFlightHistorians(), SHUTDOWN_DRAIN_MS);
		} catch (err) {
			warn("shutdown: historian drain threw:", err);
		}
		try {
			await withTimeout(awaitInFlightRecomps(), SHUTDOWN_DRAIN_MS);
		} catch (err) {
			warn("shutdown: recomp drain threw:", err);
		}
		try {
			await withTimeout(awaitInFlightDreamers(), SHUTDOWN_DRAIN_MS);
		} catch (err) {
			warn("shutdown: dreamer drain threw:", err);
		}
		try {
			for (const identity of seenDreamerProjectIdentities) {
				unregisterPiDreamerProject({ projectIdentity: identity });
			}
		} catch (err) {
			warn("shutdown: unregisterPiDreamerProject threw:", err);
		}
		// Clear per-session system-prompt adjunct caches (sticky date,
		// project docs, user profile, key files). Pi's
		// `_extensionRunner.invalidate` resets module state on session
		// swap, but on plain shutdown the maps would otherwise hold
		// their last entries. Best-effort: if sessionId can't be
		// resolved we just skip — Pi resets module state on /reload
		// anyway.
		try {
			const sm = (
				ctx as unknown as {
					sessionManager?: { getSessionId?: () => string | undefined };
				}
			).sessionManager;
			const sessionId =
				typeof sm?.getSessionId === "function" ? sm.getSessionId() : undefined;
			if (typeof sessionId === "string" && sessionId.length > 0) {
				clearPiSystemPromptSession(sessionId);
				// Drain context-handler session-keyed maps too. Without
				// this, sessions accumulate state across `session_shutdown`
				// in long-lived Pi processes that re-init the extension.
				clearContextHandlerSession(sessionId);
			}
		} catch {
			// best-effort cleanup
		}
	});

	// Pi has no `session_deleted` event, but `session_before_switch`
	// fires when the user switches to a different session within the
	// same Pi process. That's the right moment to drain caches keyed
	// by the OUTGOING session id — without this, every session swap
	// in a long-running Pi process leaks one entry per cache, and
	// after dozens of swaps the maps balloon. Cleanup here mirrors
	// OpenCode's `session.deleted` handler in `event-handler.ts`.
	pi.on("session_before_switch", (_event, ctx) => {
		try {
			const sm = (
				ctx as unknown as {
					sessionManager?: { getSessionId?: () => string | undefined };
				}
			).sessionManager;
			const outgoingSessionId =
				typeof sm?.getSessionId === "function" ? sm.getSessionId() : undefined;
			if (
				typeof outgoingSessionId === "string" &&
				outgoingSessionId.length > 0
			) {
				// Clear ONLY the in-memory per-session maps (the actual leak that
				// grows one entry per swap). Do NOT clear the durable DB m[0] cache
				// here: session_before_switch is REVERSIBLE (the user can switch
				// back), unlike OpenCode's session.deleted. The DB cache is bounded
				// (one session_meta row per session) and self-invalidates via
				// epoch/version/docs-hash checks in mustMaterializePi, so preserving
				// it lets a switch-back reuse the cached prefix instead of forcing a
				// full m[0] re-materialization (an avoidable prompt-cache bust).
				clearPiSystemPromptSession(outgoingSessionId);
				clearContextHandlerSession(outgoingSessionId);
			}
		} catch {
			// best-effort — Pi proceeds with the switch regardless
		}
	});
}
