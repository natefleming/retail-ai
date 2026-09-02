"use client";

import dynamic from "next/dynamic";

// The Console is a fully interactive client surface (streaming, localStorage,
// Vega rendering), so render it client-only to avoid SSR of browser-only code.
const Console = dynamic(
  () => import("@/components/Console").then((m) => m.Console),
  { ssr: false },
);

export default function Page() {
  return <Console />;
}
