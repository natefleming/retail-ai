import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "dao-ai Console",
  description: "Chat + live-anatomy console for dao-ai agents.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
