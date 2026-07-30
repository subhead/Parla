import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { FileText, Loader2, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { api, type ParakeetModelState, type TranscriptionSource, type WhisperModelState } from "@/lib/tauri";
import { languageLabel } from "@/lib/languages";

type Props = {
  lastWavPath: string | null;
  source: TranscriptionSource | null;
};

export function TranscribePanel({ lastWavPath, source }: Props) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState<string>("");
  const [durationMs, setDurationMs] = useState<number | null>(null);
  const [language, setLanguage] = useState<string>("auto");
  const [initialPrompt, setInitialPrompt] = useState("");
  const [nThreads, setNThreads] = useState(4);
  const [whisperModels, setWhisperModels] = useState<WhisperModelState[]>([]);
  const [parakeetModels, setParakeetModels] = useState<ParakeetModelState[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!lastWavPath || !activeModel || !ready) return;
    setBusy(true);
    setError(null);
    setText("");
    setDurationMs(null);
    try {
      const request = source?.kind === "parakeet"
        ? {
            source: "parakeet" as const,
            model_id: activeModel.id,
            language: language === "auto" ? null : language,
          }
        : {
            source: "whisper" as const,
            model_id: activeModel.id,
            language: language === "auto" ? null : language,
            initial_prompt: initialPrompt.trim() || null,
            n_threads: nThreads > 0 ? nThreads : null,
          };
      const res = await api.transcribeWav({ wav_path: lastWavPath, ...request });
      setText(res.text);
      setDurationMs(res.duration_ms);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const activeModel = useMemo(() => {
    if (source?.kind === "local") {
      return whisperModels.find((model) => model.id === source.whisper_model_id) ?? null;
    }
    if (source?.kind === "parakeet") {
      return parakeetModels.find((model) => model.id === source.parakeet_model_id) ?? null;
    }
    return null;
  }, [parakeetModels, source, whisperModels]);
  const isParakeet = source?.kind === "parakeet";
  const supportedLanguages = activeModel?.language_codes ?? [];
  const ready = Boolean(
    activeModel &&
      (isParakeet
        ? (activeModel as ParakeetModelState).downloaded &&
          (activeModel as ParakeetModelState).missing_files.length === 0
        : (activeModel as WhisperModelState).downloaded),
  );
  const disabled = !lastWavPath || !activeModel || !ready || busy || source?.kind === "cloud";

  useEffect(() => {
    Promise.all([api.listWhisperModels(), api.listParakeetModels()])
      .then(([whisper, parakeet]) => {
        setWhisperModels(whisper);
        setParakeetModels(parakeet);
      })
      .catch((e) => setError(String(e)));
  }, [source]);

  useEffect(() => {
    if (supportedLanguages.length === 0) return;
    if (!supportedLanguages.includes(language)) {
      setLanguage(supportedLanguages.includes("auto") ? "auto" : supportedLanguages[0]);
    }
  }, [language, supportedLanguages]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <CardTitle className="text-base">{t("transcribe.title")}</CardTitle>
        </div>
        <CardDescription>
          {t(isParakeet ? "transcribe.parakeetDescription" : "transcribe.description")}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between rounded-md border bg-muted/30 px-3 py-2 text-sm">
          <span className="text-muted-foreground">{t("transcribe.activeModel")}</span>
          <span className="font-medium">{activeModel?.display_name ?? t("transcribe.noSource")}</span>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="mb-1 block text-xs font-medium">{t("transcribe.language")}</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={busy}
              className="flex h-9 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm disabled:opacity-50"
            >
              {supportedLanguages.map((code) => (
                <option key={code} value={code}>
                  {code === "auto" ? t("transcribe.auto") : languageLabel(code)}
                </option>
              ))}
            </select>
          </div>
          {!isParakeet && (
            <>
              <label className="text-xs font-medium">
                {t("transcribe.initialPrompt")}
                <input value={initialPrompt} onChange={(e) => setInitialPrompt(e.target.value)} disabled={busy} className="mt-1 flex h-9 w-48 rounded-md border border-input bg-background px-3 text-sm font-normal" />
              </label>
              <label className="text-xs font-medium">
                {t("transcribe.threads")}
                <input type="number" min={1} max={64} value={nThreads} onChange={(e) => setNThreads(Number(e.target.value))} disabled={busy} className="mt-1 flex h-9 w-20 rounded-md border border-input bg-background px-3 text-sm font-normal" />
              </label>
            </>
          )}
          <Button onClick={run} disabled={disabled}>
            {busy ? <Loader2 className="animate-spin" /> : <Play />}
            {t("transcribe.transcribe")}
          </Button>
          <div className="text-xs text-muted-foreground">
            {!lastWavPath && t("transcribe.recordFirst")}
            {lastWavPath && !source && t("transcribe.noSource")}
            {lastWavPath && source?.kind === "cloud" && t("transcribe.cloudUnavailable")}
            {lastWavPath && source && !activeModel && t("transcribe.selectModel")}
            {lastWavPath && activeModel && !ready && t("transcribe.modelUnavailable")}
            {durationMs != null &&
              t("transcribe.completedIn", {
                seconds: (durationMs / 1000).toFixed(2),
              })}
          </div>
        </div>

        {text && (
          <div className="rounded-md border bg-muted/50 p-3 text-sm leading-relaxed">
            {text}
          </div>
        )}

        {error && (
          <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            {t("transcribe.errorPrefix", { message: error })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
