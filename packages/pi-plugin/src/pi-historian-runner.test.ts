import { describe, expect, it, mock } from "bun:test";
import { acquireCompartmentLease } from "@magic-context/core/features/magic-context/compartment-lease";
import {
	appendCompartments,
	getCompartments,
	getSessionFacts,
} from "@magic-context/core/features/magic-context/compartment-storage";
import { resolveProjectIdentity } from "@magic-context/core/features/magic-context/memory/project-identity";
import { getMemoriesByProject } from "@magic-context/core/features/magic-context/memory/storage-memory";
import {
	getHistorianFailureState,
	getOverflowState,
	getPendingPiCompactionMarkerState,
	getPersistedNoteNudge,
	loadProtectedTailMeta,
	recordOverflowDetected,
	reserveProtectedTailDrainTokens,
} from "@magic-context/core/features/magic-context/storage";
import { getUserMemoryCandidates } from "@magic-context/core/features/magic-context/user-memory/storage-user-memory";
import type { ProtectedTailBoundarySnapshot } from "@magic-context/core/hooks/magic-context/protected-tail-boundary";
import { closeQuietly } from "@magic-context/core/shared/sqlite-helpers";
import type { SubagentRunner } from "@magic-context/core/shared/subagent-runner";
import {
	buildPiCompactionSummary,
	runPiHistorian,
} from "./pi-historian-runner";
import { createTestDb } from "./test-utils.test";

describe("buildPiCompactionSummary", () => {
	const mk = (title: string) => ({ title, startMessage: 1, endMessage: 2 });

	it("joins all titles when at or below the cap", () => {
		const summary = buildPiCompactionSummary(["a", "b", "c", "d", "e"].map(mk));
		expect(summary).toBe("Magic Context compacted: a; b; c; d; e");
		expect(summary).not.toContain("more");
	});

	it("caps the title list and stays bounded for large compartment counts", () => {
		const many = Array.from({ length: 545 }, (_, i) => mk(`segment-${i}`));
		const summary = buildPiCompactionSummary(many);
		// Bounded: only the first 5 titles appear, plus a remainder count.
		expect(summary).toContain("Magic Context compacted 545 segments:");
		expect(summary).toContain(
			"segment-0; segment-1; segment-2; segment-3; segment-4",
		);
		expect(summary).toContain("…and 540 more");
		expect(summary).not.toContain("segment-5;");
		// Length must not scale with compartment count.
		expect(summary.length).toBeLessThan(200);
	});

	it("falls back to message range when titles are empty", () => {
		const summary = buildPiCompactionSummary([
			{ title: "  ", startMessage: 3, endMessage: 9 },
		]);
		expect(summary).toBe("Magic Context compacted messages 3-9.");
	});
});

function rawMessages(count = 12) {
	return Array.from({ length: count }, (_, index) => {
		const ordinal = index + 1;
		const isUser = ordinal % 2 === 1;
		return {
			ordinal,
			id: `m${ordinal}`,
			role: isUser ? "user" : "assistant",
			parts: [
				{
					type: "text",
					text: isUser
						? `User request ${ordinal}`
						: `Assistant response ${ordinal}`,
				},
			],
		};
	});
}

function successXml(fact = "Pi historian facts can promote to memory.") {
	return `<compartment start="1" end="2" title="Initial Pi slice">Summarized the first Pi turn.</compartment>\n<PROJECT_RULES>\n* ${fact}\n</PROJECT_RULES>`;
}

function successXmlWithUserObservation(observation: string) {
	return `${successXml()}\n<user_observations>\n* ${observation}\n</user_observations>`;
}

function makeBoundarySnapshot(
	overrides: Partial<ProtectedTailBoundarySnapshot> = {},
): ProtectedTailBoundarySnapshot {
	return {
		sessionId: "ses-historian",
		mode: "pi-trigger",
		offset: 1,
		offsetMessageId: "m1",
		protectedTailStart: 6,
		protectedTailStartMessageId: "m6",
		eligibleEndOrdinal: 6,
		eligibleEndMessageId: "m5",
		rawMessageCountAtTrigger: 12,
		rawLastMessageIdAtTrigger: "m12",
		N: 1000,
		usagePercentage: 80,
		usageInputTokens: 8000,
		usageSource: "live",
		contextLimit: 10_000,
		executeThresholdPercentage: 65,
		triggerBudget: 1000,
		priorBoundaryOrdinal: 6,
		migrationFloorActive: false,
		providerShapeVersion: "pi-folded-v1",
		cacheNamespace: "test:pi-historian",
		createdAt: 1,
		rawRangeFingerprint: "",
		trueRawEligibleTokens: 1000,
		oversizeAtomicUnit: false,
		boundaryReason: "test",
		...overrides,
	};
}

function runnerReturning(outputs: string[]): SubagentRunner {
	const run = mock(async () => {
		const text = outputs.shift() ?? "";
		return { ok: true as const, assistantText: text, durationMs: 1 };
	});
	return { harness: "pi", run } as unknown as SubagentRunner;
}

async function runHistorianWith(args: {
	outputs: string[];
	memoryEnabled?: boolean;
	autoPromote?: boolean;
	userMemoriesEnabled?: boolean;
	twoPass?: boolean;
	onPublished?: () => void;
	appendCompaction?: Parameters<typeof runPiHistorian>[0]["appendCompaction"];
	readBranchEntries?: () => unknown[];
	boundarySnapshot?: ProtectedTailBoundarySnapshot;
	refreshBoundarySnapshot?: () => ProtectedTailBoundarySnapshot;
	providerMessages?: ReturnType<typeof rawMessages>;
	beforeRun?: (db: ReturnType<typeof createTestDb>) => void;
}) {
	const db = createTestDb();
	const runner = runnerReturning([...args.outputs]);
	const holderId = "test-holder";
	expect(acquireCompartmentLease(db, "ses-historian", holderId)).not.toBeNull();
	args.beforeRun?.(db);
	await runPiHistorian({
		db,
		sessionId: "ses-historian",
		directory: process.cwd(),
		provider: { readMessages: () => args.providerMessages ?? rawMessages() },
		runner,
		historianModel: "test/model",
		historianChunkTokens: 20_000,
		twoPass: args.twoPass,
		memoryEnabled: args.memoryEnabled,
		autoPromote: args.autoPromote,
		userMemoriesEnabled: args.userMemoriesEnabled,
		onPublished: args.onPublished,
		appendCompaction: args.appendCompaction,
		readBranchEntries: args.readBranchEntries,
		boundarySnapshot: args.boundarySnapshot,
		refreshBoundarySnapshot: args.refreshBoundarySnapshot,
		compartmentLeaseHolderId: holderId,
	});
	return { db, runner };
}

describe("runPiHistorian", () => {
	it("clears emergency recovery on protected-tail-only no-op", async () => {
		const db = createTestDb();
		const runner = runnerReturning([successXml()]);
		recordOverflowDetected(db, "ses-historian", 100_000);
		appendCompartments(db, "ses-historian", [
			{
				sequence: 0,
				startMessage: 1,
				endMessage: 12,
				startMessageId: "m1",
				endMessageId: "m12",
				title: "prior",
				content: "already compacted",
			},
		]);
		try {
			await runPiHistorian({
				db,
				sessionId: "ses-historian",
				directory: process.cwd(),
				provider: { readMessages: () => rawMessages() },
				runner,
				historianModel: "test/model",
				historianChunkTokens: 20_000,
			});

			expect(runner.run).not.toHaveBeenCalled();
			expect(getOverflowState(db, "ses-historian").needsEmergencyRecovery).toBe(
				false,
			);
		} finally {
			closeQuietly(db);
		}
	});

	it("records failure when existing stored compartments fail validation", async () => {
		const db = createTestDb();
		const runner = runnerReturning([successXml()]);
		appendCompartments(db, "ses-historian", [
			{
				sequence: 0,
				startMessage: 2,
				endMessage: 3,
				startMessageId: "m2",
				endMessageId: "m3",
				title: "gap",
				content: "invalid because message 1 is missing",
			},
		]);
		try {
			await runPiHistorian({
				db,
				sessionId: "ses-historian",
				directory: process.cwd(),
				provider: { readMessages: () => rawMessages() },
				runner,
				historianModel: "test/model",
				historianChunkTokens: 20_000,
			});

			expect(runner.run).not.toHaveBeenCalled();
			expect(getHistorianFailureState(db, "ses-historian").failureCount).toBe(
				1,
			);
		} finally {
			closeQuietly(db);
		}
	});

	it("charges protected-tail drain quota by the actual Pi chunk, not the whole eligible head", async () => {
		const boundary = makeBoundarySnapshot({
			trueRawEligibleTokens: 50_000,
			N: 1_000,
			usagePercentage: 96,
		});
		const { db, runner } = await runHistorianWith({
			outputs: [successXml()],
			boundarySnapshot: boundary,
			providerMessages: rawMessages(6),
		});
		try {
			expect(runner.run.mock.calls.length).toBeGreaterThan(0);
			const charged = loadProtectedTailMeta(
				db,
				"ses-historian",
			).protectedTailDrainTokens;
			expect(charged).toBeGreaterThan(0);
			expect(charged).toBeLessThan(4_000);
		} finally {
			closeQuietly(db);
		}
	});

	it("skips when the protected-tail drain quota is exhausted", async () => {
		const boundary = makeBoundarySnapshot();
		const usable = Math.round(
			(boundary.contextLimit * boundary.executeThresholdPercentage) / 100,
		);
		const { db, runner } = await runHistorianWith({
			outputs: [successXml()],
			boundarySnapshot: boundary,
			beforeRun: (db) => {
				for (let i = 0; i < 3; i++) {
					const reservation = reserveProtectedTailDrainTokens({
						db,
						sessionId: "ses-historian",
						runId: `pre-${i}`,
						trueRawTokens: 3000,
						usagePercentage: boundary.usagePercentage,
						usable,
						perRunCap: 3000,
					});
					expect(reservation.ok).toBe(true);
				}
			},
		});
		try {
			expect(runner.run).not.toHaveBeenCalled();
			expect(
				loadProtectedTailMeta(db, "ses-historian").protectedTailDrainTokens,
			).toBe(9000);
		} finally {
			closeQuietly(db);
		}
	});

	it("rolls back reserved drain tokens when the Pi chunk is empty", async () => {
		const emptyMessages = rawMessages(4).map((message) => ({
			...message,
			parts: [],
		}));
		const { db, runner } = await runHistorianWith({
			outputs: [successXml()],
			providerMessages: emptyMessages,
			boundarySnapshot: makeBoundarySnapshot({
				protectedTailStart: 5,
				protectedTailStartMessageId: null,
				eligibleEndOrdinal: 5,
				eligibleEndMessageId: "m4",
				rawMessageCountAtTrigger: 4,
				rawLastMessageIdAtTrigger: "m4",
			}),
		});
		try {
			expect(runner.run).not.toHaveBeenCalled();
			expect(
				loadProtectedTailMeta(db, "ses-historian").protectedTailDrainTokens,
			).toBe(0);
		} finally {
			closeQuietly(db);
		}
	});

	it("refreshes a stale protected-tail snapshot and proceeds when the current boundary is runnable", async () => {
		const staleBoundary = makeBoundarySnapshot({
			rawLastMessageIdAtTrigger: "old-m12",
			rawRangeFingerprint: "stale-fingerprint",
		});
		const refreshedBoundary = makeBoundarySnapshot({
			protectedTailStart: 3,
			protectedTailStartMessageId: "m3",
			eligibleEndOrdinal: 3,
			eligibleEndMessageId: "m2",
		});
		const refreshBoundarySnapshot = mock(() => refreshedBoundary);
		const { db, runner } = await runHistorianWith({
			outputs: [successXml()],
			boundarySnapshot: staleBoundary,
			refreshBoundarySnapshot,
		});
		try {
			expect(refreshBoundarySnapshot).toHaveBeenCalledTimes(1);
			expect(runner.run).toHaveBeenCalledTimes(1);
			expect(getCompartments(db, "ses-historian")).toEqual([
				expect.objectContaining({
					sequence: 0,
					startMessage: 1,
					endMessage: 2,
					title: "Initial Pi slice",
				}),
			]);
		} finally {
			closeQuietly(db);
		}
	});

	it("keeps the stale-snapshot no-op fallback when the refreshed boundary is not runnable", async () => {
		const staleBoundary = makeBoundarySnapshot({
			rawLastMessageIdAtTrigger: "old-m12",
			rawRangeFingerprint: "stale-fingerprint",
		});
		const refreshBoundarySnapshot = mock(() =>
			makeBoundarySnapshot({
				protectedTailStart: 1,
				protectedTailStartMessageId: "m1",
				eligibleEndOrdinal: 1,
				eligibleEndMessageId: null,
				trueRawEligibleTokens: 0,
			}),
		);
		const { db, runner } = await runHistorianWith({
			outputs: [successXml()],
			boundarySnapshot: staleBoundary,
			refreshBoundarySnapshot,
		});
		try {
			expect(refreshBoundarySnapshot).toHaveBeenCalledTimes(1);
			expect(runner.run).not.toHaveBeenCalled();
			expect(getCompartments(db, "ses-historian")).toEqual([]);
		} finally {
			closeQuietly(db);
		}
	});

	it("stores userObservations as candidates (post-commit) when user memories are enabled", async () => {
		const { db } = await runHistorianWith({
			outputs: [successXmlWithUserObservation("User prefers concise answers.")],
			userMemoriesEnabled: true,
		});
		try {
			expect(getUserMemoryCandidates(db)).toEqual([
				expect.objectContaining({
					content: "User prefers concise answers.",
					sessionId: "ses-historian",
					sourceCompartmentStart: 1,
					sourceCompartmentEnd: 2,
				}),
			]);
		} finally {
			closeQuietly(db);
		}
	});
	it("does NOT store userObservations when user memories are disabled (privacy gate)", async () => {
		const { db } = await runHistorianWith({
			outputs: [successXmlWithUserObservation("User prefers concise answers.")],
			userMemoriesEnabled: false,
		});
		try {
			expect(getUserMemoryCandidates(db)).toEqual([]);
		} finally {
			closeQuietly(db);
		}
	});
	it("runs the Pi subagent, parses output, and publishes compartments and facts", async () => {
		const { db, runner } = await runHistorianWith({ outputs: [successXml()] });
		try {
			expect(runner.run).toHaveBeenCalledTimes(1);
			expect(getCompartments(db, "ses-historian")).toEqual([
				expect.objectContaining({
					sequence: 0,
					startMessage: 1,
					endMessage: 2,
					title: "Initial Pi slice",
				}),
			]);
			// v2 faithful fact lifecycle: facts are NOT written to session_facts
			// (no REPLACE). They flow to project memory via promotion.
			expect(getSessionFacts(db, "ses-historian")).toEqual([]);
			const projectPath = resolveProjectIdentity(process.cwd());
			expect(
				getMemoriesByProject(db, projectPath).map((m) => m.content),
			).toContain("Pi historian facts can promote to memory.");
		} finally {
			closeQuietly(db);
		}
	});

	it("records historian failure when first pass and repair output are invalid", async () => {
		const { db, runner } = await runHistorianWith({
			outputs: ["not xml", "still not xml"],
		});
		try {
			expect(runner.run).toHaveBeenCalledTimes(2);
			expect(getCompartments(db, "ses-historian")).toEqual([]);
			expect(getHistorianFailureState(db, "ses-historian").failureCount).toBe(
				1,
			);
		} finally {
			closeQuietly(db);
		}
	});

	it("writes Pi harness attribution on published compartments", async () => {
		const { db } = await runHistorianWith({ outputs: [successXml()] });
		try {
			const compartmentHarness = db
				.prepare("SELECT harness FROM compartments WHERE session_id = ?")
				.get("ses-historian") as { harness: string };

			expect(compartmentHarness.harness).toBe("pi");
			// v2 faithful facts: no session_facts rows are written anymore;
			// facts are promoted to project memory instead.
			const factRow = db
				.prepare("SELECT harness FROM session_facts WHERE session_id = ?")
				.get("ses-historian");
			expect(factRow).toBeNull();
		} finally {
			closeQuietly(db);
		}
	});

	it("fires note-nudge trigger and onPublished after successful publication", async () => {
		const onPublished = mock(() => undefined);
		const { db } = await runHistorianWith({
			outputs: [successXml()],
			onPublished,
		});
		try {
			expect(onPublished).toHaveBeenCalledTimes(1);
			expect(getPersistedNoteNudge(db, "ses-historian").triggerPending).toBe(
				true,
			);
		} finally {
			closeQuietly(db);
		}
	});

	it("queues a Pi-native compaction marker after publication", async () => {
		const appendCompaction = mock(() => "compact-1");
		const entries = Array.from({ length: 6 }, (_, index) => ({
			type: "message",
			id: `entry-${index + 1}`,
			message: { role: index % 2 === 0 ? "user" : "assistant" },
		}));
		const { db } = await runHistorianWith({
			outputs: [successXml()],
			appendCompaction,
			readBranchEntries: () => entries,
		});
		try {
			expect(appendCompaction).not.toHaveBeenCalled();
			expect(getPendingPiCompactionMarkerState(db, "ses-historian")).toEqual(
				expect.objectContaining({
					firstKeptEntryId: "entry-3",
					endMessageId: "m2",
					ordinal: 2,
					tokensBefore: expect.any(Number),
					summary: expect.stringContaining("Initial Pi slice"),
				}),
			);
		} finally {
			closeQuietly(db);
		}
	});

	it("promotes memories only when memoryEnabled and autoPromote allow it", async () => {
		const projectPath = resolveProjectIdentity(process.cwd());
		const allowed = await runHistorianWith({
			outputs: [successXml("Promote this Pi fact.")],
			memoryEnabled: true,
			autoPromote: true,
		});
		try {
			expect(
				getMemoriesByProject(allowed.db, projectPath).map(
					(memory) => memory.content,
				),
			).toContain("Promote this Pi fact.");
		} finally {
			closeQuietly(allowed.db);
		}

		const blocked = await runHistorianWith({
			outputs: [successXml("Do not promote this fact.")],
			memoryEnabled: false,
			autoPromote: true,
		});
		try {
			expect(getMemoriesByProject(blocked.db, projectPath)).toEqual([]);
		} finally {
			closeQuietly(blocked.db);
		}
	});

	describe("historian.two_pass", () => {
		it("does NOT run an editor pass when twoPass is false (default)", async () => {
			const { db, runner } = await runHistorianWith({
				outputs: [successXml()],
				// twoPass omitted → defaults to undefined/false
			});
			try {
				// One subagent run = first pass only.
				expect(runner.run).toHaveBeenCalledTimes(1);
			} finally {
				closeQuietly(db);
			}
		});

		it("runs the editor pass when twoPass=true and uses editor output", async () => {
			const draftXml = successXml("Draft fact only.");
			const editedXml = successXml("Edited fact replaced the draft.");
			const { db, runner } = await runHistorianWith({
				outputs: [draftXml, editedXml],
				twoPass: true,
			});
			try {
				// Two subagent runs = first pass + editor pass.
				expect(runner.run).toHaveBeenCalledTimes(2);
				expect(runner.run).toHaveBeenNthCalledWith(
					2,
					expect.not.objectContaining({ fallbackModels: expect.anything() }),
				);
				// Editor output won — the promoted fact is from the editor.
				const projectPath = resolveProjectIdentity(process.cwd());
				expect(
					getMemoriesByProject(db, projectPath).map((m) => m.content),
				).toContain("Edited fact replaced the draft.");
			} finally {
				closeQuietly(db);
			}
		});

		it("falls back to draft when editor output fails validation", async () => {
			const draftXml = successXml("Original draft fact.");
			// Editor returns garbage — validation fails, draft is preserved.
			const { db, runner } = await runHistorianWith({
				outputs: [draftXml, "not valid xml at all"],
				twoPass: true,
			});
			try {
				expect(runner.run).toHaveBeenCalledTimes(2);
				// Draft fact is promoted despite editor failure (no data loss).
				const projectPath = resolveProjectIdentity(process.cwd());
				expect(
					getMemoriesByProject(db, projectPath).map((m) => m.content),
				).toContain("Original draft fact.");
				// Compartments still persisted.
				expect(getCompartments(db, "ses-historian")).toEqual([
					expect.objectContaining({ title: "Initial Pi slice" }),
				]);
			} finally {
				closeQuietly(db);
			}
		});

		it("does NOT run editor pass when first-pass + repair both fail", async () => {
			// First-pass and repair both invalid → editor pass should be
			// skipped because there's no draft to refine.
			const { db, runner } = await runHistorianWith({
				outputs: ["not xml", "still not xml"],
				twoPass: true,
			});
			try {
				// Exactly 2 calls: first-pass + repair. NOT 3 (no editor).
				expect(runner.run).toHaveBeenCalledTimes(2);
				expect(getCompartments(db, "ses-historian")).toEqual([]);
			} finally {
				closeQuietly(db);
			}
		});
	});
});
