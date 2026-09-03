import { useMemo } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";

/**
 * Minimal, dependency-light markdown renderer for assistant answers.
 *
 * Uses `marked` (single package) + `dompurify` instead of the react-markdown /
 * remark / micromark chain, whose deep transitive tree (e.g. `zwitch`) is not
 * reliably present in the Databricks Apps internal npm proxy.
 */
export function Markdown({ children }: { children: string }) {
  const html = useMemo(() => {
    const raw = marked.parse(children ?? "", { async: false }) as string;
    return DOMPurify.sanitize(raw);
  }, [children]);

  return <div className="dao-md" dangerouslySetInnerHTML={{ __html: html }} />;
}
