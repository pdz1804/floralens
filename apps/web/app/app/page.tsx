import AppShell from "../app-shell";

// The full 7-tab functional app now lives at /app. This thin route renders the
// unchanged shell (search, pipeline, galaxy, categories, assistant, garden,
// about); the marketing landing owns "/".
export default function AppRoute() {
  return <AppShell />;
}
