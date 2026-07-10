import { afterEach, describe, expect, it, mock } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import magicContextPiExtension from "./index";
import { MAGIC_CONTEXT_PI_SUBAGENT_ENV } from "./subagent-runner";

const originalEnv = {
	MAGIC_CONTEXT_PI_SUBAGENT: process.env.MAGIC_CONTEXT_PI_SUBAGENT,
	PI_SUBAGENT_CHILD: process.env.PI_SUBAGENT_CHILD,
	XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME,
	XDG_DATA_HOME: process.env.XDG_DATA_HOME,
};

function restoreEnv() {
	for (const [key, value] of Object.entries(originalEnv)) {
		if (value === undefined) delete process.env[key];
		else process.env[key] = value;
	}
}

function isolateXdgEnv() {
	const root = mkdtempSync(join(tmpdir(), "magic-context-pi-index-test-"));
	process.env.XDG_CONFIG_HOME = join(root, "config");
	process.env.XDG_DATA_HOME = join(root, "data");
}

function createCountingPi() {
	const events: string[] = [];
	const tools: string[] = [];
	const flags: string[] = [];
	const commands: string[] = [];
	const pi = {
		on: mock((event: string) => {
			events.push(event);
		}),
		registerTool: mock((tool: { name?: string }) => {
			tools.push(tool.name ?? "<unnamed>");
		}),
		registerFlag: mock((name: string) => {
			flags.push(name);
		}),
		registerCommand: mock((name: string) => {
			commands.push(name);
		}),
		sendMessage: mock(() => undefined),
		sendUserMessage: mock(() => undefined),
	} as unknown as ExtensionAPI;
	return { pi, events, tools, flags, commands };
}

afterEach(() => {
	restoreEnv();
});

describe("Pi full extension subagent env guard", () => {
	it("no-ops before registering anything inside Magic Context Pi subagents", async () => {
		isolateXdgEnv();
		process.env[MAGIC_CONTEXT_PI_SUBAGENT_ENV] = "1";
		const registrations = createCountingPi();

		await magicContextPiExtension(registrations.pi);

		expect(registrations.events).toEqual([]);
		expect(registrations.tools).toEqual([]);
		expect(registrations.flags).toEqual([]);
		expect(registrations.commands).toEqual([]);
	});

	it("registers the full runtime when the subagent guard is absent", async () => {
		isolateXdgEnv();
		delete process.env[MAGIC_CONTEXT_PI_SUBAGENT_ENV];
		delete process.env.PI_SUBAGENT_CHILD;
		const registrations = createCountingPi();

		await magicContextPiExtension(registrations.pi);

		expect(registrations.events.length).toBeGreaterThan(0);
		expect(registrations.tools.length).toBeGreaterThan(0);
		expect(registrations.commands.length).toBeGreaterThan(0);
		expect(registrations.events).toContain("before_agent_start");
		expect(registrations.tools).toContain("ctx_search");
		expect(registrations.commands).toContain("ctx-status");
	});
});
