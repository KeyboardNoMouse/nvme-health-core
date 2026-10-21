/** @type {import('next').NextConfig} */
module.exports = {
  output: "export",                       // static HTML for GitHub Pages
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || "",
  images: { unoptimized: true },
  trailingSlash: true,
};
