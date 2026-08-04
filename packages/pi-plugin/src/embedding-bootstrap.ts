import { statSync } from "node:fs";
import {
	cortexKitProjectConfigBasePath,
	cortexKitUserConfigBasePath,
} from "@magic-context/core/config/migrate-config-location";
import {
	type EmbeddingFeatures,
	type ProjectEmbeddingRegistrationOptions,
	registerProjectEmbedding,
} from "@magic-context/core/features/magic-context/memory/embedding";
import { resolveProjectIdentityForSession } from "@magic-context/core/features/magic-context/memory/project-identity";
import type { ContextDatabase } from "@magic-context/core/features/magic-context/storage";
import {
	handleUntrustedLoad,
	isConfigLoadUntrusted,
} from "@magic-context/core/plugin/embedding-bootstrap-helpers";
import { loadPiConfigDetailed } from "./config";

interface RegistrationFingerprint {
	paths: string[];
	fingerprint: string;
}

const registrationFingerprintsByDatabase = new WeakMap<
	object,
	Map<string, RegistrationFingerprint>
>();

function configCandidatePaths(
	directory: string,
	loadedPaths: readonly string[],
): string[] {
	const projectBase = cortexKitProjectConfigBasePath(directory);
	const userBase = cortexKitUserConfigBasePath();
	return [
		`${projectBase}.jsonc`,
		`${projectBase}.json`,
		`${userBase}.jsonc`,
		`${userBase}.json`,
		...loadedPaths,
	].filter((path, index, paths) => paths.indexOf(path) === index);
}

function configFingerprint(paths: readonly string[]): string {
	return paths
		.map((path) => {
			try {
				const stat = statSync(path);
				return `${path}:${stat.size}:${stat.mtimeMs}`;
			} catch {
				return `${path}:missing`;
			}
		})
		.join("|");
}

async function ensureProjectRegisteredFromPiDirectoryWithOptions(
	directory: string,
	db: ContextDatabase,
	options: ProjectEmbeddingRegistrationOptions,
): Promise<void> {
	const projectIdentity = resolveProjectIdentityForSession(directory);
	if (!projectIdentity) return;
	let registrationFingerprints = registrationFingerprintsByDatabase.get(db);
	if (!registrationFingerprints) {
		registrationFingerprints = new Map();
		registrationFingerprintsByDatabase.set(db, registrationFingerprints);
	}
	const cached = registrationFingerprints.get(projectIdentity);
	if (cached && configFingerprint(cached.paths) === cached.fingerprint) return;

	const detailed = loadPiConfigDetailed({ cwd: directory });
	if (isConfigLoadUntrusted(detailed)) {
		handleUntrustedLoad(db, projectIdentity, directory, detailed);
		return;
	}

	const features: EmbeddingFeatures = {
		memoryEnabled: detailed.config.memory.enabled,
		gitCommitEnabled: detailed.config.memory.git_commit_indexing.enabled,
	};
	registerProjectEmbedding(
		db,
		projectIdentity,
		detailed.config.embedding,
		features,
		directory,
		options,
	);
	const fingerprintPaths = configCandidatePaths(
		directory,
		detailed.loadedFromPaths,
	);
	registrationFingerprints.set(projectIdentity, {
		paths: fingerprintPaths,
		fingerprint: configFingerprint(fingerprintPaths),
	});
}

export async function ensureProjectRegisteredFromPiDirectory(
	directory: string,
	db: ContextDatabase,
): Promise<void> {
	await ensureProjectRegisteredFromPiDirectoryWithOptions(directory, db, {
		maintenance: true,
	});
}

/**
 * Install only the process-local embedding registration for a lean child.
 *
 * This scopes `snapshot` to embedding lifecycle state: the caller still opens
 * the shared database through the normal schema-safe path because ctx_* tools
 * need a migrated, write-capable connection. What this suppresses is global
 * embedding maintenance (identity repair, ledger GC, vector repair, and
 * descriptor persistence) that child processes neither own nor need.
 */
export async function ensureProjectEmbeddingSnapshotFromPiDirectory(
	directory: string,
	db: ContextDatabase,
): Promise<void> {
	await ensureProjectRegisteredFromPiDirectoryWithOptions(directory, db, {
		maintenance: false,
	});
}
