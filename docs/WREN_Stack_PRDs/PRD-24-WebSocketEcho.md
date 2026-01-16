# PRD-24: WebSocket Echo Server

> **Sample**: `24_WebSocketEcho` | **Status**: Planned | **Depends On**: PRD-23 (SocketBasics)

## 1. Executive Summary

This PRD implements a WebSocket echo server - demonstrating the WebSocket protocol on top of raw sockets. This validates that Fidelity can handle real-world network protocols.

**Key Insight**: WebSocket is an application-level protocol. The implementation uses socket primitives (PRD-23) plus byte manipulation (regions, NativePtr) plus string handling.

## 2. Language Feature Specification

### 2.1 WebSocket Handshake

```fsharp
let handleUpgrade (client: int) (request: string) : bool =
    // Parse HTTP upgrade request
    // Extract Sec-WebSocket-Key
    // Compute accept key (SHA-1 + base64)
    // Send HTTP 101 response
```

### 2.2 WebSocket Framing

```fsharp
let readFrame (client: int) (buffer: nativeptr<byte>) : WebSocketFrame =
    // Read frame header (2+ bytes)
    // Parse opcode, mask, length
    // Read payload
    // Unmask if masked

let writeFrame (client: int) (opcode: int) (data: nativeptr<byte>) (len: int) =
    // Write frame header
    // Write payload (no mask for server->client)
```

### 2.3 Echo Loop

```fsharp
let echoLoop (client: int) =
    let region = Region.create 4
    let buffer = Region.alloc<byte> region 4096

    let mutable running = true
    while running do
        let frame = readFrame client buffer
        match frame.Opcode with
        | 0x01 ->  // Text frame
            writeFrame client 0x01 frame.Payload frame.Length
        | 0x08 ->  // Close
            running <- false
        | _ -> ()
```

## 3. FNCS Layer Implementation

### 3.1 WebSocket Types

```fsharp
// WebSocket frame structure
type WebSocketFrame = {
    Opcode: int
    IsMasked: bool
    Length: int64
    MaskKey: uint32
    Payload: nativeptr<byte>
}

// In NativeTypes.fs - records already supported
```

### 3.2 Crypto Intrinsics (Minimal)

For WebSocket handshake, we need SHA-1 and Base64:

```fsharp
| "Crypto.sha1" ->
    // nativeptr<byte> -> int -> nativeptr<byte>
    NativeType.TFun(
        NativeType.TNativePtr(env.Globals.ByteType),
        NativeType.TFun(env.Globals.IntType,
            NativeType.TNativePtr(env.Globals.ByteType)))

| "Crypto.base64Encode" ->
    // nativeptr<byte> -> int -> string
    NativeType.TFun(
        NativeType.TNativePtr(env.Globals.ByteType),
        NativeType.TFun(env.Globals.IntType, env.Globals.StringType))
```

**Note**: These can be implemented in F# using byte operations, or linked from a minimal crypto library.

## 4. Firefly/Alex Layer Implementation

### 4.1 Frame Parsing

```fsharp
let parseWebSocketFrame (buffer: nativeptr<byte>) : WebSocketFrame =
    let b0 = NativePtr.get buffer 0
    let b1 = NativePtr.get buffer 1

    let opcode = b0 &&& 0x0Fuy
    let isMasked = (b1 &&& 0x80uy) <> 0uy
    let len = b1 &&& 0x7Fuy

    // Extended length handling...
    // Mask key if masked...

    { Opcode = int opcode; ... }
```

### 4.2 Handshake Implementation

```fsharp
let webSocketAcceptKey (clientKey: string) : string =
    // Concatenate with magic GUID
    let combined = clientKey + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    // SHA-1 hash
    let hash = Crypto.sha1 (String.toBytes combined) (String.length combined)
    // Base64 encode
    Crypto.base64Encode hash 20
```

## 5. MLIR Output Specification

### 5.1 Frame Header Write

```mlir
// Write text frame with 50-byte payload
%header = llvm.alloca 2 x i8

// First byte: FIN=1, opcode=1 (text)
%b0 = arith.constant 0x81 : i8
%h0 = llvm.getelementptr %header[0]
llvm.store %b0, %h0

// Second byte: mask=0, len=50
%b1 = arith.constant 50 : i8
%h1 = llvm.getelementptr %header[1]
llvm.store %b1, %h1

// Write header
llvm.call @write(%client, %header, %c2)

// Write payload
llvm.call @write(%client, %payload, %c50)
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module WebSocketEchoSample

let handleClient (client: int) =
    let region = Region.create 4
    let buffer = Region.alloc<byte> region 4096

    // Read HTTP request
    let n = Sys.read client buffer 4096
    let request = String.fromBytes buffer n

    // Check for upgrade request
    if String.contains request "Upgrade: websocket" then
        // Perform handshake
        let key = extractWebSocketKey request
        let accept = webSocketAcceptKey key
        let response = buildUpgradeResponse accept
        Sys.write client (String.toBytes response) (String.length response)

        // Echo loop
        let mutable running = true
        while running do
            let n = Sys.read client buffer 4096
            if n > 0 then
                let frame = parseFrame buffer
                match frame.Opcode with
                | 0x01 | 0x02 ->  // Text or binary
                    sendFrame client frame.Opcode frame.Payload frame.Length
                | 0x08 ->  // Close
                    sendCloseFrame client
                    running <- false
                | 0x09 ->  // Ping
                    sendFrame client 0x0A frame.Payload frame.Length  // Pong
                | _ -> ()
            else
                running <- false

    Sys.close client

[<EntryPoint>]
let main _ =
    Console.writeln "=== WebSocket Echo Server ==="

    let sock = Sys.socket AF_INET SOCK_STREAM 0
    Sys.bind sock (SockAddr.ipv4 "0.0.0.0" 8080)
    Sys.listen sock 10

    Console.writeln "Listening on ws://localhost:8080"

    // Accept one connection for test
    let client = Sys.accept sock
    Console.writeln "Client connected"
    handleClient client

    Sys.close sock
    0
```

### 6.2 Testing

Use browser or wscat:
```bash
wscat -c ws://localhost:8080
> hello
< hello
> world
< world
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `CheckExpressions.fs` | MODIFY | Add Crypto intrinsics (optional) |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| Sample source | CREATE | WebSocket implementation in F# |

### 7.3 Alternative: F# Implementation

Most WebSocket logic can be implemented in F# using existing primitives:
- Byte manipulation: `NativePtr.get/set`
- String operations: existing intrinsics
- SHA-1/Base64: Either intrinsic or pure F# implementation

## 8. Implementation Checklist

### Phase 1: Protocol Implementation
- [ ] Implement HTTP upgrade parsing
- [ ] Implement WebSocket key calculation
- [ ] Implement frame parsing
- [ ] Implement frame writing

### Phase 2: Echo Server
- [ ] Implement handshake flow
- [ ] Implement echo loop
- [ ] Handle close frames

### Phase 3: Validation
- [ ] Sample 24 compiles
- [ ] Can connect with wscat
- [ ] Echo works correctly
- [ ] Clean close handling

## 9. Complexity Note

WebSocket is a real protocol - this sample validates that Fidelity can handle:
- HTTP parsing (strings)
- Binary protocols (byte manipulation)
- Stateful connections (loops)
- Cryptographic operations (SHA-1, Base64)

## 10. Related PRDs

- **PRD-23**: SocketBasics - Foundation
- **PRD-29-31**: MailboxProcessor - WebSocket server actor
