/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  distDir: process.env.MRTL_NEXT_DIST_DIR || ".next",
};

module.exports = nextConfig;
