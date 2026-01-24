# WebView Desktop Build Integration

> **Related Documents**:
> - [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md) - Overall architecture
> - [Architecture_Canonical.md](./Architecture_Canonical.md) - Firefly pipeline overview

---

## Overview

This document describes how Firefly orchestrates the complete build of a WebView-based desktop application. The key insight is that Firefly serves as a **unified build coordinator**, similar to how `dotnet` coordinates SAFE Stack builds or how `cargo` coordinates Rust workspaces.

A single command:
```bash
firefly build MyApp.fidproj
```

Produces a single native executable containing:
- Compiled native backend
- Embedded frontend (HTML/JS/CSS as string constant)
- Platform-specific webview bindings

---

## For .NET Developers: The Build Analogy

If you're familiar with .NET multi-project solutions, here's the mental model:

| .NET Concept | Fidelity Equivalent |
|--------------|---------------------|
| `dotnet build MySolution.sln` | `firefly build MyApp.fidproj` |
| Project references | `[desktop]` section defines frontend/backend |
| MSBuild targets | Firefly build phases |
| NuGet packages | Fidelity.Platform library + npm packages |
| `dotnet publish` | `firefly build --target <platform>` |

The difference: Firefly also orchestrates **Fable** (F# → JS) and **Vite** (bundling), producing a truly unified build.

---

## For Web Developers: What Firefly Does For You

If you're coming from web development with npm/vite, here's what changes:

**Before (typical web dev):**
```bash
# Manual multi-step process
npm run fable          # Compile F# to JS
npm run build          # Vite bundles to dist/
# Then somehow integrate with backend...
```

**After (Firefly orchestrated):**
```bash
# Single command does everything
firefly build
```

Firefly automatically:
1. Calls Fable to compile your Partas.Solid code
2. Calls Vite to bundle and optimize
3. Reads the bundled output
4. Embeds it in the native binary
5. Compiles your native backend
6. Links everything into one executable

You still write your `package.json` and `vite.config.js` normally - Firefly just invokes them at the right time.

---

## The .fidproj Schema for Desktop Applications

A desktop application uses an extended `.fidproj` schema:

```toml
[package]
name = "MyDesktopApp"
version = "0.1.0"

# Desktop-specific configuration
[desktop]
frontend = "src/Frontend"       # Path to Partas.Solid project
backend = "src/Backend"         # Path to native F# project
embed_assets = true             # Embed HTML/JS/CSS in binary

# Fable configuration (F# → JavaScript)
[desktop.fable]
output = "build/fable"          # Fable output directory
extension = ".fs.jsx"           # Output file extension
exclude = ["Partas.Solid.FablePlugin"]  # Plugins to exclude from output

# Vite configuration (bundling)
[desktop.vite]
config = "vite.config.js"       # Vite config file
output = "build/dist"           # Bundled output directory
inline = true                   # Inline all JS/CSS into HTML

# Firefly compilation settings
[compilation]
memory_model = "stack_only"
target = "native"

# Dependencies
[dependencies]
alloy = { path = "/home/user/repos/Fidelity.Platform/src" }

# Build output
[build]
output = "MyDesktopApp"         # Binary name
output_kind = "desktop"         # Triggers desktop build pipeline
```

### Schema Fields Explained

| Section | Field | Description |
|---------|-------|-------------|
| `[desktop]` | `frontend` | Path to F# project with Partas.Solid components |
| | `backend` | Path to F# project with native application code |
| | `embed_assets` | If true, HTML bundle is embedded in binary |
| `[desktop.fable]` | `output` | Where Fable writes compiled JavaScript |
| | `extension` | File extension for Fable output (`.fs.jsx` for SolidJS) |
| | `exclude` | Fable plugins to exclude from output bundle |
| `[desktop.vite]` | `config` | Path to vite.config.js |
| | `output` | Where Vite writes bundled files |
| | `inline` | Inline all assets into single HTML file |
| `[build]` | `output_kind` | `desktop` triggers the webview build pipeline |

---

## Build Phases

### Phase 1: Frontend Compilation (Fable)

Firefly invokes Fable to compile F# to JavaScript:

```bash
# What Firefly runs internally
dotnet fable {frontend_path} \
    --outDir {fable_output} \
    --extension {extension} \
    --exclude {excludes} \
    --noCache
```

**Input**: `src/Frontend/*.fs` (Partas.Solid components)
**Output**: `build/fable/*.fs.jsx` (SolidJS-compatible JavaScript)

The Partas.Solid.FablePlugin transforms `[<SolidTypeComponent>]` definitions into idiomatic SolidJS code.

### Phase 2: Asset Bundling (Vite)

Firefly invokes Vite to bundle and optimize:

```bash
# What Firefly runs internally
npm run build
# Or directly: npx vite build --config {vite_config}
```

**Input**: `build/fable/*.fs.jsx` + `index.html` + CSS
**Output**: `build/dist/index.html` (single file with inlined JS/CSS)

The `vite.config.js` should be configured for single-file output:

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

export default defineConfig({
    plugins: [solidPlugin()],
    build: {
        outDir: 'build/dist',
        // Inline all assets for single-file output
        assetsInlineLimit: 100000000,  // ~100MB limit
        cssCodeSplit: false,
        rollupOptions: {
            output: {
                inlineDynamicImports: true,
                // Single JS bundle
                manualChunks: undefined
            }
        }
    }
});
```

### Phase 3: Asset Embedding

Firefly reads the bundled HTML and generates an F# source file:

```fsharp
// Auto-generated: EmbeddedAssets.fs
module internal EmbeddedAssets

[<Literal>]
let IndexHtml = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>My Desktop App</title>
    <style>/* inlined CSS */</style>
</head>
<body>
    <div id="app"></div>
    <script>/* inlined JavaScript */</script>
</body>
</html>"""
```

The `[<Literal>]` attribute makes this a compile-time constant, ensuring it's embedded directly in the binary with no runtime overhead.

### Phase 4: Native Compilation

Firefly compiles the backend project plus the generated embedding:

```bash
# What Firefly does internally
firefly compile {backend_path} \
    --include EmbeddedAssets.fs \
    --output {output_name} \
    --target {platform}
```

**Input**: `src/Backend/*.fs` + `EmbeddedAssets.fs`
**Output**: Native executable with embedded UI

The backend's entry point uses the embedded assets:

```fsharp
// src/Backend/Main.fs
module MyApp.Main

open Fidelity.Platform
open Fidelity.Platform.Webview

let main () =
    let w = Webview.create true
    Webview.setTitle w "My Desktop App"
    Webview.setSize w 1200 800
    Webview.setHtml w EmbeddedAssets.IndexHtml  // Load embedded UI
    Webview.run w
    Webview.destroy w
    0
```

---

## Cross-Platform Building

### Building for Current Platform

```bash
firefly build
```

Produces a native executable for the current OS and architecture.

### Cross-Compilation

```bash
# Specify target platform
firefly build --target linux-x64
firefly build --target linux-arm64
firefly build --target windows-x64
firefly build --target macos-x64
firefly build --target macos-arm64
```

The frontend compilation (Fable + Vite) is platform-agnostic - JavaScript works everywhere. Only the native backend compilation differs per target.

### Build Matrix Example

For CI/CD systems:

```yaml
# GitHub Actions example
jobs:
  build:
    strategy:
      matrix:
        target: [linux-x64, windows-x64, macos-arm64]
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: firefly build --target ${{ matrix.target }}
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: app-${{ matrix.target }}
          path: ./MyDesktopApp*
```

---

## Project Structure

A typical WebView desktop project has this structure:

```
my-desktop-app/
├── my-desktop-app.fidproj      # Firefly project file
│
├── src/
│   ├── Backend/
│   │   ├── Main.fs             # Native entry point
│   │   ├── Api.fs              # Functions exposed to JS
│   │   └── Backend.fsproj      # (for IDE support only)
│   │
│   ├── Frontend/
│   │   ├── App.fs              # Root Partas.Solid component
│   │   ├── Components/
│   │   │   ├── Header.fs
│   │   │   ├── Sidebar.fs
│   │   │   └── Content.fs
│   │   └── Frontend.fsproj     # Fable project
│   │
│   └── Shared/
│       └── Types.fs            # Types shared between frontend/backend
│
├── index.html                  # Vite entry point
├── package.json                # npm dependencies
├── vite.config.js              # Vite configuration
├── tailwind.config.js          # (optional) Tailwind CSS
│
├── build/                      # Generated during build
│   ├── fable/                  # Fable output
│   ├── dist/                   # Vite output
│   └── EmbeddedAssets.fs       # Generated embedding
│
└── MyDesktopApp                # Final executable
```

### The .fsproj Files

The `Backend.fsproj` and `Frontend.fsproj` files exist for **IDE support only** (Ionide, Rider). Firefly doesn't use them - it reads the `.fidproj` directly. However, the IDE needs project files to provide IntelliSense.

**Frontend.fsproj** (for IDE):
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Fable.Core" Version="5.0.0-*" />
    <PackageReference Include="Partas.Solid" Version="2.1.3" />
    <PackageReference Include="Fable.Browser.Dom" Version="2.20.0" />
  </ItemGroup>
  <ItemGroup>
    <Compile Include="App.fs" />
    <Compile Include="Components/Header.fs" />
    <!-- ... -->
  </ItemGroup>
</Project>
```

**Backend.fsproj** (for IDE):
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="../../Fidelity.Platform/src/Fidelity.Platform.fsproj" />
  </ItemGroup>
  <ItemGroup>
    <Compile Include="Api.fs" />
    <Compile Include="Main.fs" />
  </ItemGroup>
</Project>
```

---

## npm Package Configuration

The `package.json` configures JavaScript dependencies:

```json
{
  "name": "my-desktop-app-frontend",
  "private": true,
  "type": "module",
  "scripts": {
    "fable": "dotnet fable src/Frontend -o build/fable -e .fs.jsx",
    "fable:watch": "dotnet fable watch src/Frontend -o build/fable -e .fs.jsx",
    "dev": "npm run fable && vite",
    "build": "npm run fable && vite build"
  },
  "dependencies": {
    "solid-js": "^1.9.0"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "vite-plugin-solid": "^2.8.0",
    "tailwindcss": "^3.4.0"
  }
}
```

These scripts are for **development use** (hot reloading). For production builds, Firefly invokes these commands directly.

---

## Development Workflow

### Full Rebuild

```bash
firefly build
./MyDesktopApp
```

### Hot Reload Development

For faster iteration during development:

**Terminal 1 (Frontend dev server):**
```bash
npm run dev
# Starts Vite dev server at http://localhost:5173
```

**Terminal 2 (Backend with dev URL):**
```bash
# Modify Main.fs temporarily to navigate instead of setHtml:
Webview.navigate w "http://localhost:5173"
```

This allows hot reloading of UI changes without rebuilding the native backend. For production, switch back to `setHtml` with embedded assets.

---

## Build Artifacts

After `firefly build`, you'll have:

```
my-desktop-app/
├── build/
│   ├── fable/
│   │   ├── App.fs.jsx
│   │   └── Components/*.fs.jsx
│   ├── dist/
│   │   └── index.html          # Bundled, inlined
│   └── EmbeddedAssets.fs       # Generated F# source
│
├── MyDesktopApp                # Linux/macOS executable
└── MyDesktopApp.exe            # Windows executable
```

For distribution, only the executable is needed. All assets are embedded.

---

## Error Handling

### Fable Compilation Errors

If Fable fails, Firefly reports the error and stops:

```
[FIREFLY] Phase 1: Frontend compilation (Fable)
[FABLE] error FABLE0001: Cannot find module 'Partas.Solid'
[FIREFLY] Build failed: Frontend compilation error
```

### Vite Bundling Errors

If Vite fails:

```
[FIREFLY] Phase 2: Asset bundling (Vite)
[VITE] error during build: RollupError: Could not resolve './missing.js'
[FIREFLY] Build failed: Asset bundling error
```

### Native Compilation Errors

If Firefly compilation fails:

```
[FIREFLY] Phase 4: Native compilation
[FIREFLY] error FS0001: This value is not a function
[FIREFLY] Build failed: Native compilation error
```

All errors propagate with context about which phase failed.

---

## Future: Incremental Builds

Currently, `firefly build` runs all phases. Future optimizations may include:

- **Dependency tracking**: Skip phases if inputs unchanged
- **Parallel phases**: Run Fable and backend parsing concurrently
- **Cached embeddings**: Reuse embedded assets if bundle unchanged

These optimizations are not in scope for the initial implementation.

---

## Cross-References

- [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md) - Overall architecture and IPC model
- [Architecture_Canonical.md](./Architecture_Canonical.md) - Firefly compilation pipeline
- [Native_Library_Binding_Architecture.md](./Native_Library_Binding_Architecture.md) - Platform binding patterns
