import { afterEach, describe, expect, it, mock, spyOn } from "bun:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import * as storageDb from "@magic-context/core/features/magic-context/storage-db";

const leanSubagentExtensionMock = mock((_pi: ExtensionAPI) => undefined);

mock.module("./subagent-entry", () => ({
	default: leanSubagentExtensionMock,
}));

describe("Pi full extension subagent child mode", () => {
	afterEach(() => {
		delete process.env.PI_SUBAGENT_CHILD;
		leanSubagentExtensionMock.mockClear();
		mock.restore();
	});

	it("delegates pi-subagents child processes to the lean subagent entry before opening the shared DB", async () => {
		process.env.PI_SUBAGENT_CHILD = "1";
		const openDatabaseSpy = spyOn(storageDb, "openDatabase").mockImplementation(
			() => {
				throw new Error(
					"full database startup should not run in subagent child mode",
				);
			},
		);
		const applyPragmasSpy = spyOn(
			storageDb,
			"applySqliteTuningPragmas",
		).mockImplementation(() => undefined);
		const setPragmaConfigSpy = spyOn(
			storageDb,
			"setSqlitePragmaConfig",
		).mockImplementation(() => undefined);
		const { default: registerExtension } = await import("./index");
		const pi = {} as ExtensionAPI;

		await registerExtension(pi);

		expect(leanSubagentExtensionMock).toHaveBeenCalledTimes(1);
		expect(leanSubagentExtensionMock).toHaveBeenCalledWith(pi);
		expect(openDatabaseSpy).not.toHaveBeenCalled();
		expect(applyPragmasSpy).not.toHaveBeenCalled();
		expect(setPragmaConfigSpy).not.toHaveBeenCalled();
	});
});
