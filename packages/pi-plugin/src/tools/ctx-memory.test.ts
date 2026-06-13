import { describe, expect, it } from "bun:test";
import { resolveProjectIdentity } from "@magic-context/core/features/magic-context/memory/project-identity";
import {
	getMemoryById,
	insertMemory,
} from "@magic-context/core/features/magic-context/memory/storage-memory";
import { getMemoryMutationsForRender } from "@magic-context/core/features/magic-context/storage";
import { closeQuietly } from "@magic-context/core/shared/sqlite-helpers";
import { createTestDb, fakeContext } from "../test-utils.test";
import { createCtxMemoryTool } from "./ctx-memory";

describe("createCtxMemoryTool", () => {
	it("rejects list for primary agents and allows it for dreamer agents", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const dreamer = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: true,
			});

			const ctx = fakeContext("ses-memory") as never;
			const primaryResult = await primary.execute(
				"call-1",
				{ action: "list" },
				new AbortController().signal,
				undefined,
				ctx,
			);
			const dreamerResult = await dreamer.execute(
				"call-2",
				{ action: "list" },
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(primaryResult.isError).toBe(true);
			expect(primaryResult.content[0]?.text).toBe(
				"Error: Action 'list' is not allowed in this context.",
			);
			expect(dreamerResult.isError).toBeUndefined();
			expect(dreamerResult.content[0]?.text).toBe("No active memories found.");
		} finally {
			closeQuietly(db);
		}
	});

	it("allows a primary agent to archive (no longer dreamer-only)", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;

			// write a memory as the primary agent, then archive it as the same
			// primary agent — archive replaced the old `delete` alias and is no
			// longer gated behind allowDreamerActions.
			const written = await primary.execute(
				"call-w",
				{ action: "write", category: "ARCHITECTURE", content: "Stale fact." },
				new AbortController().signal,
				undefined,
				ctx,
			);
			expect(written.isError).toBeUndefined();
			const idMatch = written.content[0]?.text?.match(/ID:\s*(\d+)/);
			const id = idMatch ? Number(idMatch[1]) : Number.NaN;
			expect(Number.isInteger(id)).toBe(true);

			const archived = await primary.execute(
				"call-a",
				{ action: "archive", ids: [id] },
				new AbortController().signal,
				undefined,
				ctx,
			);
			expect(archived.isError).toBeUndefined();
			expect(archived.content[0]?.text).toContain("Archived memory");
		} finally {
			closeQuietly(db);
		}
	});

	it("updates a foreign workspace memory under the target identity", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			// Default workspace shares CONSTRAINTS; this test exercises
			// target-identity routing, so use a shared category (foreign memory
			// visible) and verify the mutation routes under the target identity.
			db.exec(`
				INSERT INTO workspaces (id, name, created_at, updated_at) VALUES (1, 'ws', 1, 1);
				INSERT INTO workspace_members (workspace_id, project_path, display_name, display_path, added_at)
				VALUES (1, '${ownIdentity}', 'Own', '${ownIdentity}', 1),
				       (1, 'git:foreign', 'Foreign', '/foreign', 1);
			`);
			insertMemory(db, {
				projectPath: ownIdentity,
				category: "CONSTRAINTS",
				content: "Use the shared formatter.",
			});
			const foreign = insertMemory(db, {
				projectPath: "git:foreign",
				category: "CONSTRAINTS",
				content: "Old foreign directive.",
			});

			const result = await primary.execute(
				"call-u",
				{
					action: "update",
					ids: [foreign.id],
					content: "Use the shared formatter.",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(result.isError).toBeUndefined();
			expect(getMemoryById(db, foreign.id)?.content).toBe(
				"Use the shared formatter.",
			);
			expect(
				getMemoryMutationsForRender(db, ownIdentity, 0, [foreign.id]),
			).toHaveLength(0);
			expect(
				getMemoryMutationsForRender(db, "git:foreign", 0, [foreign.id]),
			).toHaveLength(1);
		} finally {
			closeQuietly(db);
		}
	});

	it("archives a foreign workspace memory under the target identity", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			// Default workspace shares CONSTRAINTS; this test exercises
			// target-identity routing, so use a shared category (foreign visible).
			db.exec(`
				INSERT INTO workspaces (id, name, created_at, updated_at) VALUES (1, 'ws', 1, 1);
				INSERT INTO workspace_members (workspace_id, project_path, display_name, display_path, added_at)
				VALUES (1, '${ownIdentity}', 'Own', '${ownIdentity}', 1),
				       (1, 'git:foreign', 'Foreign', '/foreign', 1);
			`);
			const foreign = insertMemory(db, {
				projectPath: "git:foreign",
				category: "CONSTRAINTS",
				content: "Foreign issue.",
			});

			const result = await primary.execute(
				"call-a",
				{ action: "archive", ids: [foreign.id] },
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(result.isError).toBeUndefined();
			expect(getMemoryById(db, foreign.id)?.status).toBe("archived");
			expect(
				getMemoryMutationsForRender(db, "git:foreign", 0, [foreign.id]),
			).toHaveLength(1);
		} finally {
			closeQuietly(db);
		}
	});

	it("REFUSES to archive a foreign memory in a NON-shared category (P0 parity)", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			// Workspace shares only CONSTRAINTS; a foreign ARCHITECTURE memory is
			// invisible in the render and must not be mutable by the tool either.
			db.exec(`
				INSERT INTO workspaces (id, name, created_at, updated_at, share_categories) VALUES (1, 'ws', 1, 1, '["CONSTRAINTS"]');
				INSERT INTO workspace_members (workspace_id, project_path, display_name, display_path, added_at)
				VALUES (1, '${ownIdentity}', 'Own', '${ownIdentity}', 1),
				       (1, 'git:foreign', 'Foreign', '/foreign', 1);
			`);
			const foreignHidden = insertMemory(db, {
				projectPath: "git:foreign",
				category: "ARCHITECTURE",
				content: "Foreign architecture detail not shared.",
			});

			const result = await primary.execute(
				"call-block",
				{ action: "archive", ids: [foreignHidden.id] },
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(String(result)).not.toContain("Archived memory");
			expect(getMemoryById(db, foreignHidden.id)?.status).toBe("active");
		} finally {
			closeQuietly(db);
		}
	});

	it("archives a foreign memory in a SHARED category (P0 parity)", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			db.exec(`
				INSERT INTO workspaces (id, name, created_at, updated_at, share_categories) VALUES (1, 'ws', 1, 1, '["CONSTRAINTS"]');
				INSERT INTO workspace_members (workspace_id, project_path, display_name, display_path, added_at)
				VALUES (1, '${ownIdentity}', 'Own', '${ownIdentity}', 1),
				       (1, 'git:foreign', 'Foreign', '/foreign', 1);
			`);
			const foreignShared = insertMemory(db, {
				projectPath: "git:foreign",
				category: "CONSTRAINTS",
				content: "Foreign constraint shared with this project.",
			});

			const result = await primary.execute(
				"call-ok",
				{ action: "archive", ids: [foreignShared.id] },
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(result.isError).toBeUndefined();
			expect(getMemoryById(db, foreignShared.id)?.status).toBe("archived");
		} finally {
			closeQuietly(db);
		}
	});

	it("rejects a primary-agent merge that includes another project's memory", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;

			// One memory in THIS project's identity (resolved from ctx.cwd) and
			// one under a foreign project path. Cross-identity merge is a
			// dreamer-only capability; a primary agent must get the same opaque
			// "not found" reply update/archive use (no existence oracle).
			const written = await primary.execute(
				"call-w",
				{
					action: "write",
					category: "CONSTRAINTS",
					content: "Use bun for scripts.",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);
			expect(written.isError).toBeUndefined();
			const idMatch = written.content[0]?.text?.match(/ID:\s*(\d+)/);
			const ownId = idMatch ? Number(idMatch[1]) : Number.NaN;
			expect(Number.isInteger(ownId)).toBe(true);

			const foreign = insertMemory(db, {
				projectPath: "/repo/other-project",
				category: "CONSTRAINTS",
				content: "Use bun for build scripts.",
			});

			const result = await primary.execute(
				"call-m",
				{
					action: "merge",
					ids: [ownId, foreign.id],
					content: "Use bun for all scripts in this repository.",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(result.isError).toBe(true);
			expect(result.content[0]?.text).toBe(
				`Error: Memory with ID ${foreign.id} was not found.`,
			);
			expect(getMemoryById(db, ownId)?.status).toBe("active");
			expect(getMemoryById(db, foreign.id)?.status).toBe("active");
		} finally {
			closeQuietly(db);
		}
	});

	it("REFUSES a PRIMARY merge pulling in a foreign NON-shared-category memory (P0 parity)", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			// Workspace shares only CONSTRAINTS; a foreign ARCHITECTURE memory is
			// invisible in the render and must not be mergeable by a primary agent.
			db.exec(`
				INSERT INTO workspaces (id, name, created_at, updated_at, share_categories) VALUES (1, 'ws', 1, 1, '["CONSTRAINTS"]');
				INSERT INTO workspace_members (workspace_id, project_path, display_name, display_path, added_at)
				VALUES (1, '${ownIdentity}', 'Own', '${ownIdentity}', 1),
				       (1, 'git:foreign', 'Foreign', '/foreign', 1);
			`);
			const own = insertMemory(db, {
				projectPath: ownIdentity,
				category: "ARCHITECTURE",
				content: "Own architecture detail A.",
			});
			const foreignHidden = insertMemory(db, {
				projectPath: "git:foreign",
				category: "ARCHITECTURE", // foreign, NON-shared category
				content: "Foreign architecture not shared with this project.",
			});

			const result = await primary.execute(
				"call-block",
				{
					action: "merge",
					ids: [own.id, foreignHidden.id],
					content: "Merged architecture detail.",
					category: "ARCHITECTURE",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);

			// Primary agents get the opaque "not found" reply (no existence oracle).
			expect(result.isError).toBe(true);
			expect(result.content[0]?.text).toBe(
				`Error: Memory with ID ${foreignHidden.id} was not found.`,
			);
			expect(getMemoryById(db, own.id)?.status).toBe("active");
			expect(getMemoryById(db, foreignHidden.id)?.status).toBe("active");
		} finally {
			closeQuietly(db);
		}
	});

	it("allows a PRIMARY merge of a foreign SHARED-category memory (P0 parity)", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			db.exec(`
				INSERT INTO workspaces (id, name, created_at, updated_at, share_categories) VALUES (1, 'ws', 1, 1, '["CONSTRAINTS"]');
				INSERT INTO workspace_members (workspace_id, project_path, display_name, display_path, added_at)
				VALUES (1, '${ownIdentity}', 'Own', '${ownIdentity}', 1),
				       (1, 'git:foreign', 'Foreign', '/foreign', 1);
			`);
			const own = insertMemory(db, {
				projectPath: ownIdentity,
				category: "CONSTRAINTS",
				content: "Own constraint A.",
			});
			const foreignShared = insertMemory(db, {
				projectPath: "git:foreign",
				category: "CONSTRAINTS", // shared
				content: "Foreign constraint shared with this project.",
			});

			const result = await primary.execute(
				"call-ok",
				{
					action: "merge",
					ids: [own.id, foreignShared.id],
					content: "Merged shared constraint.",
					category: "CONSTRAINTS",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);

			// New merged content matches neither source, so a FRESH canonical is
			// inserted and both sources are superseded → archived (parity with
			// OpenCode's shared-merge test).
			expect(result.isError).toBeUndefined();
			expect(getMemoryById(db, own.id)?.status).toBe("archived");
			expect(getMemoryById(db, foreignShared.id)?.status).toBe("archived");
		} finally {
			closeQuietly(db);
		}
	});

	it("REFUSES a DREAMER merge of a foreign NON-shared-category memory INSIDE a workspace (D1 parity)", async () => {
		const db = createTestDb();
		try {
			const dreamer = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: true,
			});
			const ctx = fakeContext("ses-dreamer") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			// The dreamer keeps cross-project merge OUTSIDE a workspace, but INSIDE
			// a workspace the per-category sharing policy is the user's privacy
			// boundary it must honor too.
			db.exec(`
				INSERT INTO workspaces (id, name, created_at, updated_at, share_categories) VALUES (1, 'ws', 1, 1, '["CONSTRAINTS"]');
				INSERT INTO workspace_members (workspace_id, project_path, display_name, display_path, added_at)
				VALUES (1, '${ownIdentity}', 'Own', '${ownIdentity}', 1),
				       (1, 'git:foreign', 'Foreign', '/foreign', 1);
			`);
			const own = insertMemory(db, {
				projectPath: ownIdentity,
				category: "ARCHITECTURE",
				content: "Own architecture detail D1.",
			});
			const foreignHidden = insertMemory(db, {
				projectPath: "git:foreign",
				category: "ARCHITECTURE", // foreign, NON-shared
				content: "Foreign architecture not shared with this workspace member.",
			});

			const result = await dreamer.execute(
				"call-d1-block",
				{
					action: "merge",
					ids: [own.id, foreignHidden.id],
					content: "Merged architecture detail D1.",
					category: "ARCHITECTURE",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(result.isError).toBe(true);
			expect(result.content[0]?.text).toBe(
				`Error: Memory with ID ${foreignHidden.id} is in a category not shared with this workspace member and cannot be merged.`,
			);
			expect(getMemoryById(db, own.id)?.status).toBe("active");
			expect(getMemoryById(db, foreignHidden.id)?.status).toBe("active");
		} finally {
			closeQuietly(db);
		}
	});

	it("REJECTS merging memories from DIFFERENT categories (structural guard parity)", async () => {
		const db = createTestDb();
		try {
			const dreamer = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: true,
			});
			const ctx = fakeContext("ses-dreamer") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			const arch = insertMemory(db, {
				projectPath: ownIdentity,
				category: "ARCHITECTURE",
				content: "Execute threshold capped at 80% for headroom.",
			});
			const cfg = insertMemory(db, {
				projectPath: ownIdentity,
				category: "CONFIG_VALUES",
				content: "execute_threshold_percentage accepts 20-80 scalar or map.",
			});

			const result = await dreamer.execute(
				"call-xcat",
				{
					action: "merge",
					ids: [arch.id, cfg.id],
					content: "Execute threshold stuff.",
					category: "CONFIG_VALUES",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(result.isError).toBe(true);
			expect(result.content[0]?.text).toContain("different categories");
			expect(getMemoryById(db, arch.id)?.status).toBe("active");
			expect(getMemoryById(db, cfg.id)?.status).toBe("active");
		} finally {
			closeQuietly(db);
		}
	});

	it("ALLOWS a DREAMER merge of a foreign SHARED-category memory INSIDE a workspace (D1 parity)", async () => {
		const db = createTestDb();
		try {
			const dreamer = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: true,
			});
			const ctx = fakeContext("ses-dreamer") as never;
			const ownIdentity = resolveProjectIdentity((ctx as { cwd: string }).cwd);
			db.exec(`
				INSERT INTO workspaces (id, name, created_at, updated_at, share_categories) VALUES (1, 'ws', 1, 1, '["CONSTRAINTS"]');
				INSERT INTO workspace_members (workspace_id, project_path, display_name, display_path, added_at)
				VALUES (1, '${ownIdentity}', 'Own', '${ownIdentity}', 1),
				       (1, 'git:foreign', 'Foreign', '/foreign', 1);
			`);
			const own = insertMemory(db, {
				projectPath: ownIdentity,
				category: "CONSTRAINTS",
				content: "Own constraint D1.",
			});
			const foreignShared = insertMemory(db, {
				projectPath: "git:foreign",
				category: "CONSTRAINTS", // shared
				content: "Foreign constraint shared with the workspace.",
			});

			const result = await dreamer.execute(
				"call-d1-ok",
				{
					action: "merge",
					ids: [own.id, foreignShared.id],
					content: "Merged shared constraint D1.",
					category: "CONSTRAINTS",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);

			// Fresh canonical inserted; both sources superseded → archived.
			expect(result.isError).toBeUndefined();
			expect(getMemoryById(db, own.id)?.status).toBe("archived");
			expect(getMemoryById(db, foreignShared.id)?.status).toBe("archived");
		} finally {
			closeQuietly(db);
		}
	});

	it("rejects malformed ids and duplicate merge ids for primary agents", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const projectIdentity = resolveProjectIdentity(process.cwd());
			const first = insertMemory(db, {
				projectPath: projectIdentity,
				category: "CONSTRAINTS",
				content: "Use bun for scripts.",
			});
			const second = insertMemory(db, {
				projectPath: projectIdentity,
				category: "CONSTRAINTS",
				content: "Use bun for tests.",
			});

			const malformedArchive = await primary.execute(
				"call-a",
				{ action: "archive", ids: [first.id, second.id + 0.5] },
				new AbortController().signal,
				undefined,
				ctx,
			);
			const malformedUpdate = await primary.execute(
				"call-u",
				{ action: "update", ids: [first.id + 0.5], content: "Use pnpm." },
				new AbortController().signal,
				undefined,
				ctx,
			);
			const duplicateMerge = await primary.execute(
				"call-m",
				{ action: "merge", ids: [first.id, first.id], content: "Use bun." },
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(malformedArchive.isError).toBe(true);
			expect(malformedArchive.content[0]?.text).toContain("integer memory ID");
			expect(malformedUpdate.isError).toBe(true);
			expect(malformedUpdate.content[0]?.text).toContain("integer memory ID");
			expect(duplicateMerge.isError).toBe(true);
			expect(duplicateMerge.content[0]?.text).toContain("distinct memory IDs");
			expect(getMemoryById(db, first.id)?.status).toBe("active");
			expect(getMemoryById(db, second.id)?.status).toBe("active");
		} finally {
			closeQuietly(db);
		}
	});

	it("rejects archived memories for primary update and merge", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const projectIdentity = resolveProjectIdentity(process.cwd());
			const archived = insertMemory(db, {
				projectPath: projectIdentity,
				category: "CONSTRAINTS",
				content: "Use bun for scripts.",
			});
			const active = insertMemory(db, {
				projectPath: projectIdentity,
				category: "CONSTRAINTS",
				content: "Use bun for tests.",
			});
			db.prepare("UPDATE memories SET status = 'archived' WHERE id = ?").run(
				archived.id,
			);

			const update = await primary.execute(
				"call-u",
				{ action: "update", ids: [archived.id], content: "Use pnpm." },
				new AbortController().signal,
				undefined,
				ctx,
			);
			const merge = await primary.execute(
				"call-m",
				{ action: "merge", ids: [archived.id, active.id], content: "Use bun." },
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(update.isError).toBe(true);
			expect(update.content[0]?.text).toContain("restore it before updating");
			expect(merge.isError).toBe(true);
			expect(merge.content[0]?.text).toContain("restore it before merging");
			expect(getMemoryById(db, archived.id)?.status).toBe("archived");
			expect(getMemoryById(db, active.id)?.status).toBe("active");
		} finally {
			closeQuietly(db);
		}
	});

	it("rejects archived memories for primary archive too", async () => {
		const db = createTestDb();
		try {
			const primary = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: false,
			});
			const ctx = fakeContext("ses-memory") as never;
			const projectIdentity = resolveProjectIdentity(process.cwd());
			const archived = insertMemory(db, {
				projectPath: projectIdentity,
				category: "CONSTRAINTS",
				content: "Use bun for scripts.",
			});
			db.prepare("UPDATE memories SET status = 'archived' WHERE id = ?").run(
				archived.id,
			);

			const archivedAgain = await primary.execute(
				"call-a",
				{ action: "archive", ids: [archived.id] },
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(archivedAgain.isError).toBe(true);
			expect(archivedAgain.content[0]?.text).toContain(
				"restore it before archiving",
			);
			expect(getMemoryById(db, archived.id)?.status).toBe("archived");
		} finally {
			closeQuietly(db);
		}
	});

	it("keeps dreamer able to curate archived memories during merge", async () => {
		const db = createTestDb();
		try {
			const dreamer = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: true,
			});
			const ctx = fakeContext("ses-dreamer") as never;
			const projectIdentity = resolveProjectIdentity(process.cwd());
			const archived = insertMemory(db, {
				projectPath: projectIdentity,
				category: "CONSTRAINTS",
				content: "Use bun for scripts.",
			});
			const active = insertMemory(db, {
				projectPath: projectIdentity,
				category: "CONSTRAINTS",
				content: "Use bun for tests.",
			});
			db.prepare("UPDATE memories SET status = 'archived' WHERE id = ?").run(
				archived.id,
			);

			const result = await dreamer.execute(
				"call-m",
				{
					action: "merge",
					ids: [archived.id, active.id],
					content: "Use bun for scripts.",
				},
				new AbortController().signal,
				undefined,
				ctx,
			);

			expect(result.isError).toBeUndefined();
			expect(result.content[0]?.text).toContain(
				`canonical memory [ID: ${archived.id}]`,
			);
			expect(getMemoryById(db, archived.id)?.status).toBe("active");
			expect(getMemoryById(db, active.id)?.status).toBe("archived");
		} finally {
			closeQuietly(db);
		}
	});

	it("paginates list with offset and a char-budget cap so it never overflows", async () => {
		const db = createTestDb();
		try {
			const dreamer = createCtxMemoryTool({
				db,
				memoryEnabled: true,
				embeddingEnabled: false,
				allowDreamerActions: true,
			});
			const ctx = fakeContext("ses-memory") as never;
			const signal = new AbortController().signal;

			// Write 60 large memories (~600 chars each ≈ 36KB) — above the
			// page char budget, so a single list call must page.
			const big = "X".repeat(600);
			for (let i = 0; i < 60; i++) {
				await dreamer.execute(
					"w",
					{
						action: "write",
						category: "CONSTRAINTS",
						content: `mem ${i} ${big}`,
					},
					signal,
					undefined,
					ctx,
				);
			}

			const page1 = await dreamer.execute(
				"l1",
				{ action: "list", limit: 100000 },
				signal,
				undefined,
				ctx,
			);
			const text1 = page1.content[0]?.text ?? "";
			expect(text1.length).toBeLessThan(40000);
			expect(text1).toContain("of 60 total");
			expect(text1).toContain("more.");
			expect(text1).toContain("offset=");

			// Offset past the end yields a graceful message, not a crash.
			const beyond = await dreamer.execute(
				"l2",
				{ action: "list", offset: 1000 },
				signal,
				undefined,
				ctx,
			);
			expect(beyond.content[0]?.text).toContain("use a smaller offset");
		} finally {
			closeQuietly(db);
		}
	});
});
