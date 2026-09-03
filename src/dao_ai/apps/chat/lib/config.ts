/** AppUIModel mirror (dao_ai.config.AppUIModel) + defaults + loader. */

export interface UIConfig {
  enabled: boolean;
  mode: "end_user" | "developer";
  inspector: boolean;
  session_history: boolean;
  title: string;
  subtitle: string | null;
  /** Running dao-ai version, stamped server-side. */
  version: string | null;
}

export const DEFAULT_UI_CONFIG: UIConfig = {
  enabled: true,
  mode: "end_user",
  inspector: true,
  session_history: true,
  title: "dao-ai Console",
  subtitle: null,
  version: null,
};

/** Fetch the deploy-injected AppUIModel from the server, falling back to
 * defaults for any field the server omits (zero-config => full Console). */
export async function loadUIConfig(): Promise<UIConfig> {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return DEFAULT_UI_CONFIG;
    const raw = (await res.json()) as Partial<UIConfig> | null;
    // Drop null/undefined so an unset server field (e.g. title: null) falls
    // back to the default instead of overriding it.
    const cleaned = Object.fromEntries(
      Object.entries(raw ?? {}).filter(([, v]) => v != null),
    );
    return { ...DEFAULT_UI_CONFIG, ...cleaned };
  } catch {
    return DEFAULT_UI_CONFIG;
  }
}
