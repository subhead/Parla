import type { TFunction } from "i18next";

export type ModelCatalogKind = "whisper" | "parakeet" | "gguf" | "cloud";

function catalogId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_");
}

export function localizeModel(t: TFunction, kind: ModelCatalogKind, id: string, displayName: string, notes: string, imported = false) {
  const key = `modelCatalog.${kind}.${catalogId(id)}`;
  return {
    displayName: imported ? t("modelCatalog.imported.name", { name: displayName }) : t(`${key}.name`, { defaultValue: displayName }),
    notes: imported ? t("modelCatalog.imported.notes", { name: notes }) : t(`${key}.notes`, { defaultValue: notes }),
  };
}

export function localizedModelName(t: TFunction, kind: ModelCatalogKind, id: string, displayName: string, notes = "", imported = false): string {
  return localizeModel(t, kind, id, displayName, notes, imported).displayName;
}
