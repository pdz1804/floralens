/** @type {import('next').NextConfig} */
// Proxy API + health to the FastAPI backend so the browser stays same-origin
// (no CORS changes needed). Override with FLORALENS_API_BASE.
const API_BASE = process.env.FLORALENS_API_BASE || "http://127.0.0.1:8100";

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_BASE}/api/:path*` },
      { source: "/health", destination: `${API_BASE}/health` },
    ];
  },
};

module.exports = nextConfig;
