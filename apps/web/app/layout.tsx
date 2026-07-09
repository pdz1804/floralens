import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FloraLens — Flower Similarity Search",
  description: "Paste a flower photo and find visually similar specimens with confidence scores.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
