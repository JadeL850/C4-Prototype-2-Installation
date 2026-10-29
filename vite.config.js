// =============================================================================
// File:    vite.config.js
// Project: MediaPipe Demos — MDes Prototyping, CCA
// Authors: Copilot
//          Thomas J McLeish
// License: MIT — see LICENSE in the root of this repository
// =============================================================================

import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 5500,
    strictPort: true,
    open: "/"
  }
});