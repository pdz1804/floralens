import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import "./globals.css";

// Editorial serif for the wordmark, result names, and section titles.
const playfair = Playfair_Display({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-serif",
  display: "swap",
});

// Clean, highly legible sans for body copy and UI controls.
const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FloraLens — Flower Similarity Search",
  description: "Paste a flower photo and find visually similar specimens with confidence scores.",
};

// Runs synchronously before first paint so the stored theme choice is applied
// to <html data-theme> with no flash of the wrong theme. Default is "system"
// (which follows prefers-color-scheme via CSS).
const themeScript = `(function(){try{var t=localStorage.getItem('floralens-theme');document.documentElement.setAttribute('data-theme',t==='light'||t==='dark'||t==='system'?t:'system');}catch(e){document.documentElement.setAttribute('data-theme','system');}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${playfair.variable} ${inter.variable}`}
      data-theme="system"
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
