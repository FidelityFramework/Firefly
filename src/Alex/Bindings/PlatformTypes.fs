/// PlatformTypes - Basic platform identification types
///
/// These types are extracted to break circular dependencies:
/// - PlatformConfig needs OSFamily/Architecture
/// - PSGZipper needs PlatformConfig
/// - BindingTypes needs PSGZipper
///
/// By putting OSFamily/Architecture here, all can import without cycles.
module Alex.Bindings.PlatformTypes

// ===================================================================
// Target Platform Identification
// ===================================================================

/// Operating system family
type OSFamily =
    | Linux
    | Windows
    | MacOS
    | FreeBSD
    | BareMetal
    | WASM

/// Processor architecture
type Architecture =
    | X86_64
    | ARM64
    | ARM32_Thumb
    | RISCV64
    | RISCV32
    | WASM32

/// Complete platform identification
type TargetPlatform = {
    OS: OSFamily
    Arch: Architecture
    Triple: string
    Features: Set<string>
}

module TargetPlatform =
    let parseTriple (triple: string) : TargetPlatform option =
        let parts = triple.ToLowerInvariant().Split('-')
        let archOsOpt =
            match parts with
            | [| arch; _vendor; os |] -> Some (arch, os)
            | [| arch; _vendor; os; _env |] -> Some (arch, os)
            | _ -> None
        match archOsOpt with
        | None -> None
        | Some (arch, os) ->
            let architecture =
                match arch with
                | "x86_64" | "amd64" -> Some X86_64
                | "aarch64" | "arm64" -> Some ARM64
                | a when a.StartsWith("armv7") || a.StartsWith("thumb") -> Some ARM32_Thumb
                | "riscv64" -> Some RISCV64
                | "riscv32" -> Some RISCV32
                | "wasm32" -> Some WASM32
                | _ -> None

            let osFamily =
                match os with
                | o when o.StartsWith("linux") -> Some Linux
                | o when o.StartsWith("windows") -> Some Windows
                | o when o.StartsWith("darwin") || o.StartsWith("macos") -> Some MacOS
                | o when o.StartsWith("freebsd") -> Some FreeBSD
                | "none" | "unknown" when arch.StartsWith("thumb") -> Some BareMetal
                | "unknown" when arch = "wasm32" -> Some WASM
                | _ -> None

            match architecture, osFamily with
            | Some arch, Some os ->
                Some { OS = os; Arch = arch; Triple = triple; Features = Set.empty }
            | _ -> None

    let linux_x86_64 = { OS = Linux; Arch = X86_64; Triple = "x86_64-unknown-linux-gnu"; Features = Set.empty }
    let windows_x86_64 = { OS = Windows; Arch = X86_64; Triple = "x86_64-pc-windows-msvc"; Features = Set.empty }
    let macos_x86_64 = { OS = MacOS; Arch = X86_64; Triple = "x86_64-apple-darwin"; Features = Set.empty }
    let macos_arm64 = { OS = MacOS; Arch = ARM64; Triple = "aarch64-apple-darwin"; Features = Set.empty }
    let linux_arm64 = { OS = Linux; Arch = ARM64; Triple = "aarch64-unknown-linux-gnu"; Features = Set.empty }
    let linux_riscv64 = { OS = Linux; Arch = RISCV64; Triple = "riscv64-unknown-linux-gnu"; Features = Set.empty }
    let baremetal_thumb = { OS = BareMetal; Arch = ARM32_Thumb; Triple = "thumbv7em-none-eabi"; Features = Set.empty }
    let wasm32 = { OS = WASM; Arch = WASM32; Triple = "wasm32-unknown-unknown"; Features = Set.empty }

    let detectHost () : TargetPlatform =
        let os =
            if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Linux) then Linux
            elif System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows) then Windows
            elif System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.OSX) then MacOS
            else failwith "detectHost: Unsupported OS"

        let arch =
            match System.Runtime.InteropServices.RuntimeInformation.OSArchitecture with
            | System.Runtime.InteropServices.Architecture.X64 -> X86_64
            | System.Runtime.InteropServices.Architecture.Arm64 -> ARM64
            | System.Runtime.InteropServices.Architecture.Arm -> ARM32_Thumb
            | other -> failwithf "detectHost: Unsupported architecture %A" other

        let triple =
            match os, arch with
            | Linux, X86_64 -> "x86_64-unknown-linux-gnu"
            | Linux, ARM64 -> "aarch64-unknown-linux-gnu"
            | Windows, X86_64 -> "x86_64-pc-windows-msvc"
            | MacOS, X86_64 -> "x86_64-apple-darwin"
            | MacOS, ARM64 -> "aarch64-apple-darwin"
            | _ -> "x86_64-unknown-linux-gnu"

        { OS = os; Arch = arch; Triple = triple; Features = Set.empty }
