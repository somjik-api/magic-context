import {
	type EmbeddingFeatures,
	type ProjectEmbeddingRegistrationOptions,
	registerProjectEmbeddingAndMaybeWipe,
} from "@magic-context/core/features/magic-context/memory/embedding";
import { resolveProjectIdentity } from "@magic-context/core/features/magic-context/memory/project-identity";
import type { ContextDatabase } from "@magic-context/core/features/magic-context/storage";
import {
	handleUntrustedLoad,
	isConfigLoadUntrusted,
} from "@magic-context/core/plugin/embedding-bootstrap-helpers";
import { loadPiConfigDetailed } from "./config";

async function ensureProjectRegisteredFromPiDirectoryWithOptions(
	directory: string,
	db: ContextDatabase,
	options: ProjectEmbeddingRegistrationOptions,
): Promise<void> {
	const projectIdentity = resolveProjectIdentity(directory);

	const detailed = loadPiConfigDetailed({ cwd: directory });
	if (isConfigLoadUntrusted(detailed)) {
		handleUntrustedLoad(db, projectIdentity, directory, detailed);
		return;
	}

	const features: EmbeddingFeatures = {
		memoryEnabled: detailed.config.memory.enabled,
		gitCommitEnabled: detailed.config.memory.git_commit_indexing.enabled,
	};
	registerProjectEmbeddingAndMaybeWipe(
		db,
		projectIdentity,
		detailed.config.embedding,
		features,
		directory,
		options,
	);
}

export async function ensureProjectRegisteredFromPiDirectory(
	directory: string,
	db: ContextDatabase,
): Promise<void> {
	await ensureProjectRegisteredFromPiDirectoryWithOptions(directory, db, {
		maintenance: true,
	});
}

export async function ensureProjectEmbeddingSnapshotFromPiDirectory(
	directory: string,
	db: ContextDatabase,
): Promise<void> {
	await ensureProjectRegisteredFromPiDirectoryWithOptions(directory, db, {
		maintenance: false,
	});
}
