# PRD-23: Socket I/O Basics

> **Sample**: `23_SocketBasics` | **Status**: Planned | **Depends On**: PRD-17-19 (Async), PRD-20-22 (Regions)

## 1. Executive Summary

This PRD introduces socket I/O - the foundation for network programming. Sockets enable TCP/UDP communication using OS syscalls directly, without managed runtime overhead.

**Key Insight**: Sockets are file descriptors. The same `Sys.read`/`Sys.write` primitives work for sockets as for files. This PRD adds socket-specific syscalls: `socket`, `bind`, `listen`, `accept`, `connect`.

## 2. Language Feature Specification

### 2.1 Socket Creation

```fsharp
let sock = Sys.socket AF_INET SOCK_STREAM 0
```

Creates a TCP socket.

### 2.2 Server Pattern

```fsharp
let server () =
    let sock = Sys.socket AF_INET SOCK_STREAM 0
    Sys.bind sock (SockAddr.ipv4 "0.0.0.0" 8080)
    Sys.listen sock 10

    let client = Sys.accept sock
    let buffer = Region.alloc<byte> region 1024
    let n = Sys.read client buffer 1024
    // ... process ...
    Sys.close client
    Sys.close sock
```

### 2.3 Client Pattern

```fsharp
let client () =
    let sock = Sys.socket AF_INET SOCK_STREAM 0
    Sys.connect sock (SockAddr.ipv4 "127.0.0.1" 8080)
    Sys.write sock data len
    Sys.close sock
```

## 3. FNCS Layer Implementation

### 3.1 Socket Intrinsics

```fsharp
// In CheckExpressions.fs
| "Sys.socket" ->
    // int -> int -> int -> int (domain, type, protocol -> fd)
    NativeType.TFun(env.Globals.IntType,
        NativeType.TFun(env.Globals.IntType,
            NativeType.TFun(env.Globals.IntType, env.Globals.IntType)))

| "Sys.bind" ->
    // int -> SockAddr -> int (fd, addr -> result)
    NativeType.TFun(env.Globals.IntType,
        NativeType.TFun(NativeType.TSockAddr, env.Globals.IntType))

| "Sys.listen" ->
    // int -> int -> int (fd, backlog -> result)
    NativeType.TFun(env.Globals.IntType,
        NativeType.TFun(env.Globals.IntType, env.Globals.IntType))

| "Sys.accept" ->
    // int -> int (server_fd -> client_fd)
    NativeType.TFun(env.Globals.IntType, env.Globals.IntType)

| "Sys.connect" ->
    // int -> SockAddr -> int (fd, addr -> result)
    NativeType.TFun(env.Globals.IntType,
        NativeType.TFun(NativeType.TSockAddr, env.Globals.IntType))

| "Sys.close" ->
    // int -> int (fd -> result)
    NativeType.TFun(env.Globals.IntType, env.Globals.IntType)
```

### 3.2 SockAddr Type

```fsharp
// In NativeTypes.fs
| TSockAddr  // Opaque socket address

// Helper module
| "SockAddr.ipv4" ->
    // string -> int -> SockAddr
    NativeType.TFun(env.Globals.StringType,
        NativeType.TFun(env.Globals.IntType, NativeType.TSockAddr))
```

### 3.3 Constants

```fsharp
| "AF_INET" -> NativeType.TInt, 2
| "SOCK_STREAM" -> NativeType.TInt, 1
| "SOCK_DGRAM" -> NativeType.TInt, 2
```

## 4. Firefly/Alex Layer Implementation

### 4.1 Platform Bindings

**Linux x86_64**:
| Syscall | Number |
|---------|--------|
| socket | 41 |
| bind | 49 |
| listen | 50 |
| accept | 43 |
| connect | 42 |
| close | 3 |
| read | 0 |
| write | 1 |

### 4.2 SockAddr Construction

```fsharp
let witnessSockAddrIpv4 z ipStrSSA portSSA =
    // struct sockaddr_in { sin_family, sin_port, sin_addr, sin_zero }
    let addrSSA = freshSSA ()

    emit $"  %%{addrSSA} = llvm.alloca 1 x !sockaddr_in"

    // sin_family = AF_INET
    emit $"  %%family_ptr = llvm.getelementptr %%{addrSSA}[0, 0]"
    emit "  llvm.store %c2, %family_ptr"  // AF_INET = 2

    // sin_port = htons(port)
    emit $"  %%port_net = llvm.call @htons(%%{portSSA})"
    emit $"  %%port_ptr = llvm.getelementptr %%{addrSSA}[0, 1]"
    emit "  llvm.store %port_net, %port_ptr"

    // sin_addr = inet_pton(ip)
    emit $"  %%addr_ptr = llvm.getelementptr %%{addrSSA}[0, 2]"
    emit $"  llvm.call @inet_pton(i32 2, %%{ipStrSSA}, %%addr_ptr)"

    TRValue { SSA = addrSSA; Type = TSockAddr }
```

### 4.3 Syscall Witnesses

```fsharp
let witnessSocket z domainSSA typeSSA protoSSA =
    let fdSSA = freshSSA ()
    emit $"  %%{fdSSA} = llvm.call @socket(%%{domainSSA}, %%{typeSSA}, %%{protoSSA})"
    TRValue { SSA = fdSSA; Type = TInt }

let witnessBind z fdSSA addrSSA =
    let resultSSA = freshSSA ()
    emit $"  %%{resultSSA} = llvm.call @bind(%%{fdSSA}, %%{addrSSA}, i32 16)"
    TRValue { SSA = resultSSA; Type = TInt }
```

## 5. MLIR Output Specification

### 5.1 SockAddr Struct

```mlir
!sockaddr_in = !llvm.struct<(
    i16,     // sin_family
    i16,     // sin_port (network byte order)
    i32,     // sin_addr (network byte order)
    array<8 x i8>  // sin_zero
)>
```

### 5.2 Socket Creation

```mlir
// let sock = Sys.socket AF_INET SOCK_STREAM 0
%domain = arith.constant 2 : i32   // AF_INET
%type = arith.constant 1 : i32     // SOCK_STREAM
%proto = arith.constant 0 : i32
%sock = llvm.call @socket(%domain, %type, %proto) : (i32, i32, i32) -> i32
```

### 5.3 Bind and Listen

```mlir
// Sys.bind sock addr
%addr = ...  // sockaddr_in
%bind_result = llvm.call @bind(%sock, %addr, %c16) : (i32, !llvm.ptr, i32) -> i32

// Sys.listen sock 10
%listen_result = llvm.call @listen(%sock, %c10) : (i32, i32) -> i32
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module SocketBasicsSample

[<EntryPoint>]
let main _ =
    Console.writeln "=== Socket Basics Test ==="

    // Create a TCP socket
    let sock = Sys.socket AF_INET SOCK_STREAM 0
    if sock < 0 then
        Console.writeln "Failed to create socket"
        1
    else
        Console.writeln "Socket created successfully"

        // Bind to port 8080
        let addr = SockAddr.ipv4 "0.0.0.0" 8080
        let bindResult = Sys.bind sock addr
        if bindResult < 0 then
            Console.writeln "Failed to bind"
        else
            Console.writeln "Bound to port 8080"

            // Listen
            let listenResult = Sys.listen sock 10
            Console.writeln "Listening..."

            // For test: just verify we can set up, don't actually accept
            Console.writeln "Server setup complete"

        Sys.close sock
        0
```

### 6.2 Expected Output

```
=== Socket Basics Test ===
Socket created successfully
Bound to port 8080
Listening...
Server setup complete
```

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TSockAddr type |
| `CheckExpressions.fs` | MODIFY | Add socket syscall intrinsics |
| `NativeGlobals.fs` | MODIFY | Add socket constants |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `src/Alex/Bindings/Socket_Linux_x86_64.fs` | CREATE | Linux socket syscall bindings |
| `src/Alex/Witnesses/SocketWitness.fs` | CREATE | Socket operation witnesses |

## 8. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add TSockAddr type
- [ ] Add socket intrinsics
- [ ] Add socket constants

### Phase 2: Alex Implementation
- [ ] Create socket platform bindings
- [ ] Implement SockAddr construction
- [ ] Implement socket syscall witnesses

### Phase 3: Validation
- [ ] Sample 23 compiles without errors
- [ ] Sample 23 creates and binds socket
- [ ] Samples 01-22 still pass

## 9. Error Handling

Socket operations return -1 on error with errno set. For now, error handling is manual checking. Future: Result<int, SocketError> wrappers.

## 10. Related PRDs

- **PRD-24**: WebSocket - Protocol on top of sockets
- **PRD-17-19**: Async - Async socket operations
- **PRD-29-31**: MailboxProcessor - Network actors
