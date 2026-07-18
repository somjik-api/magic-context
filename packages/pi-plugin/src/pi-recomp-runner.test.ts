/// <reference types="bun-types" />

import { expect, test } from "bun:test";
import {
	readRawSessionMessages,
	setRawMessageProvider,
} from "@magic-context/core/hooks/magic-context/read-session-chunk";
import { awaitInFlightRecomps, spawnPiRecompRun } from "./pi-recomp-runner";

test("detached recomp keeps its full-history provider across a concurrent context provider replacement", async () => {
	const sessionId = "ses-recomp-provider-scope";
	const fullMessages = [
		{
			ordinal: 1,
			id: "full-1",
			role: "user" as const,
			parts: [{ type: "text" as const, text: "full" }],
		},
	];
	const suffixMessages = [
		{
			ordinal: 2,
			id: "suffix-2",
			role: "user" as const,
			parts: [{ type: "text" as const, text: "suffix" }],
		},
	];
	let observed = [] as typeof fullMessages;
	let unregisterSuffix: (() => void) | undefined;

	spawnPiRecompRun({
		sessionId,
		provider: { readMessages: () => fullMessages },
		onStatusChange: () => {},
		work: async () => {
			await Promise.resolve();
			unregisterSuffix = setRawMessageProvider(sessionId, {
				readMessages: () => suffixMessages,
			});
			observed = readRawSessionMessages(sessionId) as typeof fullMessages;
		},
	});
	await awaitInFlightRecomps();

	expect(observed).toEqual(fullMessages);
	expect(readRawSessionMessages(sessionId)).toEqual(suffixMessages);
	unregisterSuffix?.();
});
