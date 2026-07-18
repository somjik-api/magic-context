import type { Memory } from "../../features/magic-context/memory";
import { LIST_PAGE_CHAR_BUDGET } from "./constants";

export interface MemoryListPage {
    /** Memories selected for this page (already offset-sliced by the caller). */
    pageMemories: Memory[];
    /** Total memories in the filtered set across all pages. */
    totalCount: number;
    /** Zero-based offset of the first row in `pageMemories`. */
    offset: number;
    /** Optional category filter that every follow-up page must preserve. */
    category?: string;
}

interface MemoryListRow {
    id: string;
    category: string;
    status: string;
    verification: string;
    updated: string;
    content: string;
}

const CONTENT_TRUNCATION_MARKER = "… [content truncated to fit page budget]";

function renderMemoryRows(page: MemoryListPage, rows: MemoryListRow[]): string {
    const { pageMemories, totalCount, offset } = page;
    const headers = {
        id: "ID",
        category: "CATEGORY",
        status: "STATUS",
        verification: "VERIFY",
        updated: "UPDATED",
        content: "CONTENT",
    };
    const widths = {
        id: Math.max(headers.id.length, ...rows.map((row) => row.id.length)),
        category: Math.max(headers.category.length, ...rows.map((row) => row.category.length)),
        status: Math.max(headers.status.length, ...rows.map((row) => row.status.length)),
        verification: Math.max(
            headers.verification.length,
            ...rows.map((row) => row.verification.length),
        ),
        updated: Math.max(headers.updated.length, ...rows.map((row) => row.updated.length)),
    };
    const formatRow = (row: MemoryListRow | typeof headers) =>
        [
            row.id.padEnd(widths.id),
            row.category.padEnd(widths.category),
            row.status.padEnd(widths.status),
            row.verification.padEnd(widths.verification),
            row.updated.padEnd(widths.updated),
            row.content,
        ].join(" | ");

    const shownEnd = offset + rows.length;
    const hasMore = shownEnd < totalCount;
    const categoryArgument = page.category ? `, category=${JSON.stringify(page.category)}` : "";
    const footer = hasMore
        ? `\n\n… ${totalCount - shownEnd} more. Fetch the next page with ctx_memory(action="list"${categoryArgument}, offset=${shownEnd}${
              pageMemories.length > rows.length ? "" : `, limit=${rows.length}`
          }).`
        : "";

    return [
        `Showing memories ${offset + 1}-${shownEnd} of ${totalCount} total.`,
        "",
        formatRow(headers),
        [
            "-".repeat(widths.id),
            "-".repeat(widths.category),
            "-".repeat(widths.status),
            "-".repeat(widths.verification),
            "-".repeat(widths.updated),
            "-------",
        ].join("-+-"),
        ...rows.map(formatRow),
    ]
        .join("\n")
        .concat(footer);
}

export function formatMemoryList(page: MemoryListPage): string {
    const { pageMemories, totalCount, offset } = page;
    if (totalCount === 0) {
        return "No active memories found.";
    }
    if (pageMemories.length === 0) {
        return `No memories at offset ${offset}. Total is ${totalCount}; use a smaller offset.`;
    }

    const allRows: MemoryListRow[] = pageMemories.map((memory) => ({
        id: String(memory.id),
        category: memory.category,
        status: memory.status,
        verification: memory.verificationStatus,
        updated: new Date(memory.updatedAt).toISOString(),
        content: memory.content.replace(/\s+/g, " ").trim(),
    }));

    // Keep the cheap estimate to avoid formatting an unbounded number of rows,
    // then enforce the budget against the exact final rendering below.
    const rows: MemoryListRow[] = [];
    let estimatedChars = 0;
    for (const row of allRows) {
        const rowChars = row.content.length + 80;
        if (rows.length > 0 && estimatedChars + rowChars > LIST_PAGE_CHAR_BUDGET) {
            break;
        }
        rows.push(row);
        estimatedChars += rowChars;
    }

    let output = renderMemoryRows(page, rows);
    while (output.length > LIST_PAGE_CHAR_BUDGET && rows.length > 1) {
        rows.pop();
        output = renderMemoryRows(page, rows);
    }
    if (output.length <= LIST_PAGE_CHAR_BUDGET) {
        return output;
    }

    // A single stored memory can itself exceed the page budget. Preserve its
    // metadata and a deterministic content prefix, but never emit an oversized
    // tool result. Stored content remains untouched and is still searchable.
    const first = rows[0];
    const markerOnlyOutput = renderMemoryRows(page, [
        { ...first, content: CONTENT_TRUNCATION_MARKER },
    ]);
    const contentPrefixChars = Math.max(0, LIST_PAGE_CHAR_BUDGET - markerOnlyOutput.length);
    output = renderMemoryRows(page, [
        {
            ...first,
            content: `${first.content.slice(0, contentPrefixChars)}${CONTENT_TRUNCATION_MARKER}`,
        },
    ]);

    // Metadata fields are bounded by the storage schema, so the exact calculation
    // above should hit the limit at most. Keep a final defensive clamp so this
    // function's public hard-cap contract cannot regress if metadata grows later.
    return output.slice(0, LIST_PAGE_CHAR_BUDGET);
}
