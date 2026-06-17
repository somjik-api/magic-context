import { describe, expect, it } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	getProjectEmbeddings,
	peekProjectEmbeddings,
	resetEmbeddingCacheForTests,
} from "@magic-context/core/features/magic-context/memory/embedding-cache";
import { resolveProjectIdentity } from "@magic-context/core/features/magic-context/memory/project-identity";
import { closeQuietly } from "@magic-context/core/shared/sqlite-helpers";
import {
	ensureProjectEmbeddingSnapshotFromPiDirectory,
	ensureProjectRegisteredFromPiDirectory,
} from "./embedding-bootstrap";
import { createTestDb } from "./test-utils.test";

describe("ensureProjectRegisteredFromPiDirectory", () => {
	it("registers a subagent embedding snapshot without maintenance", async () => {
		const db = createTestDb();
		const oldHome = process.env.HOME;
		const directory = mkdtempSync(join(tmpdir(), "pi-embedding-subagent-"));
		const fakeHome = mkdtempSync(join(tmpdir(), "pi-embedding-home-"));
		process.env.HOME = fakeHome;
		resetEmbeddingCacheForTests();
		try {
			const projectIdentity = resolveProjectIdentity(directory);

			await ensureProjectEmbeddingSnapshotFromPiDirectory(directory, db);

			const cached = getProjectEmbeddings(db, projectIdentity);
			cached.set(7, new Float32Array([7]));
			expect(peekProjectEmbeddings(projectIdentity)).toBe(cached);
		} finally {
			resetEmbeddingCacheForTests();
			if (oldHome === undefined) {
				delete process.env.HOME;
			} else {
				process.env.HOME = oldHome;
			}
			closeQuietly(db);
		}
	});

	it("preserves the embedding cache across consecutive identical registrations", async () => {
		const db = createTestDb();
		const oldHome = process.env.HOME;
		const directory = mkdtempSync(join(tmpdir(), "pi-embedding-bootstrap-"));
		const fakeHome = mkdtempSync(join(tmpdir(), "pi-embedding-home-"));
		process.env.HOME = fakeHome;
		resetEmbeddingCacheForTests();
		try {
			const projectIdentity = resolveProjectIdentity(directory);

			await ensureProjectRegisteredFromPiDirectory(directory, db);
			const cached = getProjectEmbeddings(db, projectIdentity);
			cached.set(42, new Float32Array([1, 2, 3]));

			await ensureProjectRegisteredFromPiDirectory(directory, db);

			expect(peekProjectEmbeddings(projectIdentity)).toBe(cached);
			expect(peekProjectEmbeddings(projectIdentity)?.get(42)).toEqual(
				new Float32Array([1, 2, 3]),
			);
		} finally {
			resetEmbeddingCacheForTests();
			if (oldHome === undefined) {
				delete process.env.HOME;
			} else {
				process.env.HOME = oldHome;
			}
			closeQuietly(db);
		}
	});
});
