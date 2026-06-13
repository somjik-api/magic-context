import { describe, expect, it } from "bun:test";
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
