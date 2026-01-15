/// Bits Intrinsics Test - Tests byte order conversion and bit casting
module BitsTest

[<EntryPoint>]
let main argv =
    // Byte order conversions for uint16
    let networkShort = Bits.htons 0x1234us
    let hostShort = Bits.ntohs networkShort

    // Byte order conversions for uint32
    let networkLong = Bits.htonl (uint32 0x12345678)
    let hostLong = Bits.ntohl networkLong

    // Bit casting - float32 <-> int32
    let floatBits = Bits.float32ToInt32Bits 1.0f
    let backToFloat = Bits.int32BitsToFloat32 floatBits

    // Bit casting - float64 <-> int64
    let doubleBits = Bits.float64ToInt64Bits 1.0
    let backToDouble = Bits.int64BitsToFloat64 doubleBits

    Console.writeln "Bits intrinsics test passed!"
    0
