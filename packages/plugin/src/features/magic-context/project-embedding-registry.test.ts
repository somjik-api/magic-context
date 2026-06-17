import { afterEach, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { EmbeddingConfig } from "../../config/schema/magic-context";
import { Database } from "../../shared/sqlite";
import {
    chunkCanonicalText,
    loadCompartmentChunkEmbeddingsForSearch,
    replaceCompartmentChunkEmbeddings,
} from "./compartment-chunk-embedding";
import { appendCompartments, getCompartments } from "./compartment-storage";
import type { EmbeddingProvider } from "./memory/embedding-provider";
import { insertMemory } from "./memory/storage-memory";
import { getStoredModelId, saveEmbedding } from "./memory/storage-memory-embeddings";
import {
    _resetProjectEmbeddingRegistryForTests,
    _setTestProviderFactoryForProject,
    embedSessionCompartmentChunks,
    embedTextForProject,
    embedUnembeddedCompartmentChunksForProject,
    getProjectEmbeddingSnapshot,
    registerProjectEmbeddingAndMaybeWipe,
    registerProjectInObservationMode,
    sweepAllRegisteredProjects,
} from "./project-embedding-registry";
import { recordSessionProjectIdentity } from "./session-project-storage";
import { closeDatabase, openDatabase } from "./storage";

class FakeEmbeddingProvider implements EmbeddingProvider {
    readonly modelId: string;
    disposed = false;

    constructor(modelId: string) {
        this.modelId = modelId;
    }

    async initialize(): Promise<boolean> {
        return true;
    }

    async embed(text: string): Promise<Float32Array> {
        return new Float32Array([text.length, this.modelId.length]);
    }

    async embedBatch(texts: string[]): Promise<Float32Array[]> {
        return texts.map((text) => new Float32Array([text.length, this.modelId.length]));
    }

    async dispose(): Promise<void> {
        this.disposed = true;
    }

    isLoaded(): boolean {
        return true;
    }
}

function localConfig(model: string, maxInputTokens?: number): EmbeddingConfig {
    return {
        provider: "local",
        model,
        ...(maxInputTokens !== undefined ? { max_input_tokens: maxInputTokens } : {}),
    };
}

function seedCompartmentWithFts(
    db: NonNullable<ReturnType<typeof openDatabase>>,
    sessionId: string,
): number {
    appendCompartments(db, sessionId, [
        {
            sequence: 0,
            startMessage: 1,
            endMessage: 2,
            startMessageId: "u1",
            endMessageId: "a2",
            title: "Hydraulic backpressure",
            content: "P1 content",
            p1: "P1 content",
        },
    ]);
    db.prepare(
        "INSERT INTO message_history_fts (session_id, message_ordinal, message_id, role, content) VALUES (?, ?, ?, ?, ?)",
    ).run(sessionId, 1, `${sessionId}-u1`, "user", "How do we avoid saturating the queue?");
    db.prepare(
        "INSERT INTO message_history_fts (session_id, message_ordinal, message_id, role, content) VALUES (?, ?, ?, ?, ?)",
    ).run(sessionId, 2, `${sessionId}-a2`, "assistant", "Use backpressure and bounded drains.");
    return getCompartments(db, sessionId)[0].id;
}

function seedManyCompartmentsWithFts(
    db: NonNullable<ReturnType<typeof openDatabase>>,
    sessionId: string,
    count: number,
): void {
    for (let i = 0; i < count; i++) {
        const start = i * 2 + 1;
        const end = start + 1;
        appendCompartments(db, sessionId, [
            {
                sequence: i,
                startMessage: start,
                endMessage: end,
                startMessageId: `u${start}`,
                endMessageId: `a${end}`,
                title: `Compartment ${i}`,
                content: `P1 content ${i}`,
                p1: `P1 content ${i}`,
            },
        ]);
        db.prepare(
            "INSERT INTO message_history_fts (session_id, message_ordinal, message_id, role, content) VALUES (?, ?, ?, ?, ?)",
        ).run(sessionId, start, `${sessionId}-u${start}`, "user", `Question ${i}?`);
        db.prepare(
            "INSERT INTO message_history_fts (session_id, message_ordinal, message_id, role, content) VALUES (?, ?, ?, ?, ?)",
        ).run(sessionId, end, `${sessionId}-a${end}`, "assistant", `Answer ${i}.`);
    }
}

describe("project embedding registry", () => {
    const tempDirs: string[] = [];
    const originalXdgDataHome = process.env.XDG_DATA_HOME;

    function useTempDb() {
        const dir = mkdtempSync(join(tmpdir(), "project-embedding-registry-"));
        tempDirs.push(dir);
        process.env.XDG_DATA_HOME = dir;
        return openDatabase();
    }

    afterEach(() => {
        _resetProjectEmbeddingRegistryForTests();
        closeDatabase();
        process.env.XDG_DATA_HOME = originalXdgDataHome;
        for (const dir of tempDirs) {
            try {
                rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
            } catch {
                /* Ignore EBUSY on Windows */
            }
        }
        tempDirs.length = 0;
    });

    it("supports snapshot-only registration without running storage maintenance", async () => {
        _setTestProviderFactoryForProject(
            (config) =>
                new FakeEmbeddingProvider(config.provider === "local" ? config.model : "off"),
        );
        const schemaLessDb = new Database(":memory:");
        try {
            const snapshot = registerProjectEmbeddingAndMaybeWipe(
                schemaLessDb,
                "git:subagent",
                localConfig("model-subagent"),
                { memoryEnabled: true, gitCommitEnabled: true },
                "/tmp/subagent",
                { maintenance: false },
            );
            const vector = await embedTextForProject("git:subagent", "hello");

            expect(snapshot.projectIdentity).toBe("git:subagent");
            expect(snapshot.enabled).toBe(true);
            expect(snapshot.gitCommitEnabled).toBe(true);
            expect(vector?.vector[1]).toBe("model-subagent".length);
        } finally {
            schemaLessDb.close();
        }
    });

    it("keeps independent snapshots and providers for two projects in one process", async () => {
        _setTestProviderFactoryForProject(
            (config) =>
                new FakeEmbeddingProvider(config.provider === "local" ? config.model : "off"),
        );

        registerProjectEmbeddingAndMaybeWipe(
            useTempDb(),
            "git:project-a",
            localConfig("model-a"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/a",
        );
        registerProjectEmbeddingAndMaybeWipe(
            openDatabase(),
            "git:project-b",
            localConfig("model-b-long"),
            { memoryEnabled: true, gitCommitEnabled: true },
            "/tmp/b",
        );

        const first = getProjectEmbeddingSnapshot("git:project-a");
        const second = getProjectEmbeddingSnapshot("git:project-b");
        const firstVector = await embedTextForProject("git:project-a", "hello");
        const secondVector = await embedTextForProject("git:project-b", "hello");

        expect(first?.projectIdentity).toBe("git:project-a");
        expect(first?.sourceDirectory).toBe("/tmp/a");
        expect(first?.enabled).toBe(true);
        expect(first?.gitCommitEnabled).toBe(false);
        expect(second?.projectIdentity).toBe("git:project-b");
        expect(second?.sourceDirectory).toBe("/tmp/b");
        expect(second?.enabled).toBe(true);
        expect(second?.gitCommitEnabled).toBe(true);
        expect(firstVector?.modelId).not.toBe(secondVector?.modelId);
        expect(firstVector?.vector[1]).not.toBe(secondVector?.vector[1]);
    });

    it("uses observation:<sha> fingerprints and disables runtime reads for corrupt first-time config", () => {
        const snapshot = registerProjectInObservationMode(
            useTempDb(),
            "git:corrupt",
            "/tmp/corrupt",
            { provider: "off" },
            "embedding config parse failed",
        );

        expect(snapshot.runtimeFingerprint).toMatch(/^observation:[0-9a-f]+$/);
        expect(snapshot.enabled).toBe(false);
        expect(snapshot.gitCommitEnabled).toBe(false);
        expect(snapshot.modelId).toBe("off");
        expect(getProjectEmbeddingSnapshot("git:corrupt")?.runtimeFingerprint).toBe(
            snapshot.runtimeFingerprint,
        );
    });

    it("invalidates stale embedding results when registration changes during an in-flight call", async () => {
        let release: (() => void) | undefined;
        _setTestProviderFactoryForProject(
            (config) =>
                new (class extends FakeEmbeddingProvider {
                    async embed(): Promise<Float32Array> {
                        await new Promise<void>((resolve) => {
                            release = resolve;
                        });
                        return new Float32Array([1, 2]);
                    }
                })(config.provider === "local" ? config.model : "off"),
        );

        registerProjectEmbeddingAndMaybeWipe(
            useTempDb(),
            "git:project",
            localConfig("model-a"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/project",
        );
        const inFlight = embedTextForProject("git:project", "hello");
        registerProjectEmbeddingAndMaybeWipe(
            openDatabase(),
            "git:project",
            localConfig("model-b"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/project",
        );

        release?.();

        expect(await inFlight).toBeNull();
    });

    it("wipes stale compartment chunk embeddings on provider change", () => {
        const db = useTempDb();
        const compartmentId = seedCompartmentWithFts(db, "ses-wipe");
        const windows = chunkCanonicalText("[1] U: hello", 1, 1, 10_000);
        replaceCompartmentChunkEmbeddings(
            db,
            windows.map((window) => ({
                compartmentId,
                sessionId: "ses-wipe",
                projectPath: "git:wipe",
                window,
                modelId: "stale:model",
                vector: new Float32Array([1, 0]),
            })),
        );

        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:wipe",
            localConfig("model-b"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/wipe",
        );

        expect(loadCompartmentChunkEmbeddingsForSearch(db, "ses-wipe", "git:wipe")).toHaveLength(0);
    });

    it("backfill drains missing compartment chunks and is idempotent", async () => {
        let batchCalls = 0;
        _setTestProviderFactoryForProject(
            (config) =>
                new (class extends FakeEmbeddingProvider {
                    override async embedBatch(texts: string[]): Promise<Float32Array[]> {
                        batchCalls += 1;
                        return super.embedBatch(texts);
                    }
                })(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        seedCompartmentWithFts(db, "ses-backfill");
        recordSessionProjectIdentity(db, "ses-backfill", "git:backfill");
        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:backfill",
            localConfig("model-a"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/backfill",
        );

        const first = await sweepAllRegisteredProjects(db, 5);
        expect(first.chunksEmbedded).toBe(1);
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-backfill", "git:backfill"),
        ).toHaveLength(1);

        const second = await sweepAllRegisteredProjects(db, 5);
        expect(second.chunksEmbedded).toBe(0);
        expect(batchCalls).toBe(1);
    });

    it("keeps passive chunk backfill scoped to the caller project", async () => {
        _setTestProviderFactoryForProject(
            (config) =>
                new FakeEmbeddingProvider(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        seedCompartmentWithFts(db, "ses-project-a");
        seedCompartmentWithFts(db, "ses-project-b");
        recordSessionProjectIdentity(db, "ses-project-a", "git:project-a");
        recordSessionProjectIdentity(db, "ses-project-b", "git:project-b");
        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:project-a",
            localConfig("model-a"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/project-a",
        );

        const embedded = await embedUnembeddedCompartmentChunksForProject(db, "git:project-a");

        expect(embedded).toBe(1);
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-project-a", "git:project-a"),
        ).toHaveLength(1);
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-project-b", "git:project-a"),
        ).toHaveLength(0);
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-project-b", "git:project-b"),
        ).toHaveLength(0);
    });

    it("repairs chunk rows stamped with a different project than their session owner", () => {
        const db = useTempDb();
        const compartmentId = seedCompartmentWithFts(db, "ses-repair");
        const windows = chunkCanonicalText("[1] U: hello", 1, 1, 10_000);
        replaceCompartmentChunkEmbeddings(
            db,
            windows.map((window) => ({
                compartmentId,
                sessionId: "ses-repair",
                projectPath: "git:wrong",
                window,
                modelId: "chunk:model",
                vector: new Float32Array([1, 0]),
            })),
        );

        recordSessionProjectIdentity(db, "ses-repair", "git:right");

        expect(loadCompartmentChunkEmbeddingsForSearch(db, "ses-repair", "git:wrong")).toHaveLength(
            0,
        );
        expect(loadCompartmentChunkEmbeddingsForSearch(db, "ses-repair", "git:right")).toHaveLength(
            1,
        );
    });

    it("does not backfill compartment chunks when memory is disabled", async () => {
        _setTestProviderFactoryForProject(
            (config) =>
                new FakeEmbeddingProvider(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        seedCompartmentWithFts(db, "ses-memory-off");
        recordSessionProjectIdentity(db, "ses-memory-off", "git:off");
        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:off",
            localConfig("model-a"),
            { memoryEnabled: false, gitCommitEnabled: false },
            "/tmp/off",
        );

        const result = await sweepAllRegisteredProjects(db, 5);
        expect(result.chunksEmbedded).toBe(0);
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-memory-off", "git:off"),
        ).toHaveLength(0);
    });

    it("re-embeds chunks but preserves memory vectors when max_input_tokens changes", async () => {
        _setTestProviderFactoryForProject(
            (config) =>
                new FakeEmbeddingProvider(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        seedCompartmentWithFts(db, "ses-window");
        const firstSnapshot = registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:window",
            localConfig("model-a", 1),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/window",
        );
        const memory = insertMemory(db, {
            projectPath: "git:window",
            category: "CONSTRAINTS",
            content: "Preserve provider-scoped memory vectors across chunk window changes.",
        });
        saveEmbedding(db, memory.id, new Float32Array([1, 2]), firstSnapshot.modelId);

        const first = await embedSessionCompartmentChunks(db, "git:window", "ses-window");
        expect(first.status).toBe("done");
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-window", "git:window"),
        ).not.toHaveLength(0);

        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:window",
            localConfig("model-a", 10_000),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/window",
        );

        expect(getStoredModelId(db, "git:window")).toBe(firstSnapshot.modelId);
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-window", "git:window"),
        ).toHaveLength(0);

        const second = await embedSessionCompartmentChunks(db, "git:window", "ses-window");
        expect(second.status).toBe("done");
        expect(second.embedded).toBe(1);
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-window", "git:window"),
        ).toHaveLength(1);
    });

    it("embedSessionCompartmentChunks drains a whole session and reports progress", async () => {
        _setTestProviderFactoryForProject(
            (config) =>
                new FakeEmbeddingProvider(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        // Three compartments in the session, plus one in a DIFFERENT session
        // that must NOT be touched (session-scoped, not project-scoped).
        seedManyCompartmentsWithFts(db, "ses-embed", 3);
        seedCompartmentWithFts(db, "ses-other");
        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:embed",
            localConfig("model-a"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/embed",
        );

        const progress: Array<{ embedded: number; total: number }> = [];
        const outcome = await embedSessionCompartmentChunks(db, "git:embed", "ses-embed", {
            batchSize: 1,
            onProgress: (p) => progress.push({ ...p }),
        });

        expect(outcome.status).toBe("done");
        expect(outcome.embedded).toBe(3);
        expect(outcome.total).toBe(3);
        // Progress is monotonic and ends at total; the only session embedded is ses-embed.
        expect(progress.at(-1)).toEqual({ embedded: 3, total: 3 });
        expect(
            loadCompartmentChunkEmbeddingsForSearch(db, "ses-embed", "git:embed").length,
        ).toBeGreaterThanOrEqual(3);
        expect(loadCompartmentChunkEmbeddingsForSearch(db, "ses-other", "git:embed")).toHaveLength(
            0,
        );

        // Idempotent: a second run finds nothing.
        const again = await embedSessionCompartmentChunks(db, "git:embed", "ses-embed");
        expect(again.status).toBe("nothing");
        expect(again.total).toBe(0);
    });

    it("embedSessionCompartmentChunks reports stalled when the provider returns null vectors", async () => {
        // Provider that yields null for every text → no compartment can persist.
        _setTestProviderFactoryForProject(
            (config) =>
                new (class extends FakeEmbeddingProvider {
                    override async embedBatch(texts: string[]): Promise<Float32Array[]> {
                        return texts.map(() => null as unknown as Float32Array);
                    }
                })(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        seedManyCompartmentsWithFts(db, "ses-stall", 2);
        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:stall",
            localConfig("model-a"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/stall",
        );

        const outcome = await embedSessionCompartmentChunks(db, "git:stall", "ses-stall", {
            batchSize: 1,
        });
        expect(outcome.status).toBe("stalled");
        expect(outcome.embedded).toBe(0);
        if (outcome.status === "stalled") {
            expect(outcome.remaining).toBe(2);
        }
    });

    it("embedSessionCompartmentChunks caps windows per provider call across compartments", async () => {
        const callWindowCounts: number[] = [];
        _setTestProviderFactoryForProject(
            (config) =>
                new (class extends FakeEmbeddingProvider {
                    override async embedBatch(texts: string[]): Promise<Float32Array[]> {
                        callWindowCounts.push(texts.length);
                        return super.embedBatch(texts);
                    }
                })(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        // 20 single-window compartments + a large batchSize so the drain selects
        // many at once — the per-call WINDOW cap (16) must split them across
        // multiple provider calls even though they fit in one candidate query.
        const sessionId = "ses-manycomp";
        seedManyCompartmentsWithFts(db, sessionId, 20);
        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:manycomp",
            localConfig("model-a"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/manycomp",
        );

        const outcome = await embedSessionCompartmentChunks(db, "git:manycomp", sessionId, {
            batchSize: 20,
        });
        expect(outcome.status).toBe("done");
        expect(outcome.embedded).toBe(20);
        // No single provider call exceeded the window cap, and 20 one-window
        // compartments required more than one call (20 > 16).
        expect(callWindowCounts.length).toBeGreaterThan(1);
        expect(Math.max(...callWindowCounts)).toBeLessThanOrEqual(16);
    });

    it("embedSessionCompartmentChunks returns disabled when memory is off", async () => {
        _setTestProviderFactoryForProject(
            (config) =>
                new FakeEmbeddingProvider(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        seedCompartmentWithFts(db, "ses-off");
        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:embed-off",
            localConfig("model-a"),
            { memoryEnabled: false, gitCommitEnabled: false },
            "/tmp/embed-off",
        );

        const outcome = await embedSessionCompartmentChunks(db, "git:embed-off", "ses-off");
        expect(outcome.status).toBe("disabled");
        expect(outcome.embedded).toBe(0);
    });

    it("embedSessionCompartmentChunks aborts cleanly on signal", async () => {
        _setTestProviderFactoryForProject(
            (config) =>
                new FakeEmbeddingProvider(config.provider === "local" ? config.model : "off"),
        );
        const db = useTempDb();
        seedManyCompartmentsWithFts(db, "ses-abort", 4);
        registerProjectEmbeddingAndMaybeWipe(
            db,
            "git:abort",
            localConfig("model-a"),
            { memoryEnabled: true, gitCommitEnabled: false },
            "/tmp/abort",
        );

        const controller = new AbortController();
        const outcome = await embedSessionCompartmentChunks(db, "git:abort", "ses-abort", {
            batchSize: 1,
            signal: controller.signal,
            onProgress: ({ embedded }) => {
                if (embedded >= 2) controller.abort();
            },
        });

        expect(outcome.status).toBe("aborted");
        expect(outcome.embedded).toBeGreaterThanOrEqual(2);
        expect(outcome.embedded).toBeLessThan(4);
    });
});
