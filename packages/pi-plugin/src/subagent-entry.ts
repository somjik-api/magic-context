/**
 * Magic Context — Pi subagent extension entry.
 *
 * This is a lean extension entry loaded ONLY in child Pi processes
 * spawned by `PiSubagentRunner` (sidekick, dreamer, historian, etc.).
 * It registers Magic Context's tool surface for the subagent — nothing
 * else.
 *
 * Why this exists: Magic Context's main entry (`./index.ts`) registers
 * historian, dreamer, transform pipeline, nudges, system-prompt
 * injection, command palette, and agent_end cleanup. Loading that full
 * extension in subagents would:
 *   1. Cause recursion (subagent's own historian fires, spawning another
 *      subagent, etc.) — exactly the failure mode `--no-extensions`
 *      was originally added to avoid.
 *   2. Waste startup time on resource discovery and timer wiring.
 *   3. Inject prompt content the subagent prompt doesn't expect (key
 *      files, project docs, user profile, session history, etc.).
 *
 * What this entry registers for `--no-session` children:
 *   - `ctx_search` — read-only search over shared memories/messages/git
 *   - `ctx_memory` — dreamer only; sidekick is retrieval-only and uses ctx_search
 *
 * Session-scoped `ctx_note`/`ctx_expand` stay omitted because hidden child
 * sessions have no useful transcript or parent note id to target.
 *
 * Recursion guard: this entry never wires `pi.on("context", ...)`,
 * `pi.on("before_agent_start", ...)`, or any other event that could
 * trigger historian/dreamer/transform pipelines. Subagents only get the
 * tool surface — that's it.
 *
 * How parent passes this entry to the child:
 *   pi --print --no-extensions \
 *     --extension /absolute/path/to/dist/subagent-entry.js \
 *     [other flags...]
 *
 * `--no-extensions` skips Pi's discovered-extensions scan but still
 * loads explicit `--extension` paths. The result: subagent gets the scoped
 * Magic Context tools without full-extension overhead or recursion risk.
 *
 * Tool/action allowlists via Pi flags:
 *   --magic-context-dreamer-actions  Register ctx_memory with the dreamer
 *                                     action surface. Off by default; sidekick
 *                                     receives ctx_search only.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { ContextDatabase } from "@magic-context/core/features/magic-context/storage";
import { openDatabase } from "@magic-context/core/features/magic-context/storage-db";
import { setHarness } from "@magic-context/core/shared/harness";
import { log } from "@magic-context/core/shared/logger";
import { loadPiConfig } from "./config";
import { ensureProjectEmbeddingSnapshotFromPiDirectory } from "./embedding-bootstrap";
import { registerMagicContextTools } from "./tools";

const SUBAGENT_DREAMER_ACTIONS_FLAG = "magic-context-dreamer-actions";

let openedDb: ContextDatabase | undefined;

export default function magicContextSubagentExtension(pi: ExtensionAPI): void {
	// Mark this Pi process as a Magic Context subagent in the shared
	// harness state. session-scoped writes from any code path that
	// reaches the shared core will tag rows with harness='pi' the same
	// way the main extension does — but in practice subagents shouldn't
	// be writing session-scoped state at all.
	setHarness("pi");

	pi.registerFlag(SUBAGENT_DREAMER_ACTIONS_FLAG, {
		description:
			"Register ctx_memory with dreamer actions for Magic Context subagents.",
		type: "boolean",
		default: false,
	});

	pi.on("session_start", async () => {
		try {
			const db = openDatabase();
			if (!db) {
				throw new Error(
					"storage open failed; refusing to start without Magic Context tools",
				);
			}
			openedDb = db;

			// Load shared config so embedding settings + memory enabled
			// flag match the parent's runtime. Subagent doesn't honor
			// historian/dreamer/sidekick blocks at all (those are
			// parent-only concerns).
			const directory = process.cwd();
			const { config: cfg } = loadPiConfig({ cwd: directory });
			await ensureProjectEmbeddingSnapshotFromPiDirectory(directory, db);
			const dreamerActionsEnabled =
				pi.getFlag(SUBAGENT_DREAMER_ACTIONS_FLAG) === true;

			registerMagicContextTools(pi, {
				db,
				ensureProjectRegistered: ensureProjectEmbeddingSnapshotFromPiDirectory,
				// Sidekick is retrieval-only and consumes untrusted /ctx-aug prompt text,
				// so only dreamer subagents register ctx_memory in child processes.
				memoryToolEnabled: dreamerActionsEnabled,
				allowDreamerActions: dreamerActionsEnabled,
				// `--no-session` children resolve getSessionId() to the ephemeral
				// child session, so session-scoped ctx_note/ctx_expand would write
				// orphaned notes / expand an empty transcript. Drop them; keep ctx_search.
				sessionScopedToolsDisabled: true,
			});

			log(
				`[pi-subagent] registered tools: ctx_search${dreamerActionsEnabled ? ", ctx_memory" : ""}` +
					` (ctx_note/ctx_expand omitted: --no-session child;` +
					` memory=${cfg.memory.enabled}, embedding=${cfg.embedding.provider !== "off"},` +
					` git_commits=${cfg.memory.git_commit_indexing.enabled}, dreamer_actions=${dreamerActionsEnabled})`,
			);
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			log(`[pi-subagent] startup failed: ${message}`);
			process.exitCode = 1;
			throw err;
		}
	});

	pi.on("session_shutdown", () => {
		if (openedDb) {
			try {
				openedDb.close();
			} catch {
				// ignore close errors during shutdown
			}
			openedDb = undefined;
		}
	});
}
