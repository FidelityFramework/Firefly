/// Platform Helper Functions - MLIR function definitions for common operations
///
/// ARCHITECTURAL FOUNDATION:
/// Instead of expanding complex operations inline at every call site,
/// we define them once as func.func definitions that get emitted at module level.
/// This follows the "platform as library" pattern used by libc but without runtime overhead.
///
/// These helpers use the func and scf dialects and will be lowered to LLVM by mlir-opt.
module Alex.Bindings.PlatformHelpers

open Alex.Traversal.MLIRZipper

// ═══════════════════════════════════════════════════════════════════════════
// Helper Names (used for registration and call emission)
// ═══════════════════════════════════════════════════════════════════════════

[<Literal>]
let ParseIntHelper = "fidelity_parse_int"

[<Literal>]
let ParseFloatHelper = "fidelity_parse_float"

[<Literal>]
let StringContainsCharHelper = "fidelity_string_contains_char"

// ═══════════════════════════════════════════════════════════════════════════
// Helper Bodies (MLIR func.func definitions)
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a string to int64
/// Input: fat string (!llvm.struct<(ptr, i64)>)
/// Output: i64
let parseIntBody = """
func.func @fidelity_parse_int(%str: !llvm.struct<(ptr, i64)>) -> i64 {
  // Extract pointer and length
  %ptr = llvm.extractvalue %str[0] : !llvm.struct<(ptr, i64)>
  %len = llvm.extractvalue %str[1] : !llvm.struct<(ptr, i64)>

  // Constants
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c10 = arith.constant 10 : i64
  %c48 = arith.constant 48 : i64
  %c45_i8 = arith.constant 45 : i8

  // Check if first char is '-'
  %first_char = llvm.load %ptr : !llvm.ptr -> i8
  %is_neg = arith.cmpi eq, %first_char, %c45_i8 : i8

  // Starting position: 1 if negative, 0 if positive
  %start_pos = arith.select %is_neg, %c1, %c0 : i64

  // Parse digits with scf.while loop
  %result:2 = scf.while (%val = %c0, %pos = %start_pos) : (i64, i64) -> (i64, i64) {
    %in_bounds = arith.cmpi slt, %pos, %len : i64
    scf.condition(%in_bounds) %val, %pos : i64, i64
  } do {
  ^bb0(%val_arg: i64, %pos_arg: i64):
    %char_ptr = llvm.getelementptr %ptr[%pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char = llvm.load %char_ptr : !llvm.ptr -> i8
    %char_i64 = arith.extui %char : i8 to i64
    %digit = arith.subi %char_i64, %c48 : i64
    %val_times_10 = arith.muli %val_arg, %c10 : i64
    %new_val = arith.addi %val_times_10, %digit : i64
    %new_pos = arith.addi %pos_arg, %c1 : i64
    scf.yield %new_val, %new_pos : i64, i64
  }

  // Apply sign: if negative, negate result
  %negated = arith.subi %c0, %result#0 : i64
  %final = arith.select %is_neg, %negated, %result#0 : i64

  return %final : i64
}
"""

/// Parse a string to float64
/// Input: fat string (!llvm.struct<(ptr, i64)>)
/// Output: f64
let parseFloatBody = """
func.func @fidelity_parse_float(%str: !llvm.struct<(ptr, i64)>) -> f64 {
  // Extract pointer and length
  %ptr = llvm.extractvalue %str[0] : !llvm.struct<(ptr, i64)>
  %len = llvm.extractvalue %str[1] : !llvm.struct<(ptr, i64)>

  // Constants
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c10_i64 = arith.constant 10 : i64
  %c48 = arith.constant 48 : i64
  %c45_i8 = arith.constant 45 : i8
  %c46_i8 = arith.constant 46 : i8
  %c0_f64 = arith.constant 0.0 : f64
  %c1_f64 = arith.constant 1.0 : f64
  %c10_f64 = arith.constant 10.0 : f64

  // Check if first char is '-'
  %first_char = llvm.load %ptr : !llvm.ptr -> i8
  %is_neg = arith.cmpi eq, %first_char, %c45_i8 : i8
  %start_pos = arith.select %is_neg, %c1_i64, %c0_i64 : i64

  // Parse integer part (before decimal point)
  %int_result:2 = scf.while (%val = %c0_i64, %pos = %start_pos) : (i64, i64) -> (i64, i64) {
    %in_bounds = arith.cmpi slt, %pos, %len : i64
    %char_ptr_check = llvm.getelementptr %ptr[%pos] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char_check = llvm.load %char_ptr_check : !llvm.ptr -> i8
    %not_dot = arith.cmpi ne, %char_check, %c46_i8 : i8
    %continue = arith.andi %in_bounds, %not_dot : i1
    scf.condition(%continue) %val, %pos : i64, i64
  } do {
  ^bb0(%val_arg: i64, %pos_arg: i64):
    %char_ptr = llvm.getelementptr %ptr[%pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char = llvm.load %char_ptr : !llvm.ptr -> i8
    %char_i64 = arith.extui %char : i8 to i64
    %digit = arith.subi %char_i64, %c48 : i64
    %val_times_10 = arith.muli %val_arg, %c10_i64 : i64
    %new_val = arith.addi %val_times_10, %digit : i64
    %new_pos = arith.addi %pos_arg, %c1_i64 : i64
    scf.yield %new_val, %new_pos : i64, i64
  }

  // Convert integer part to float
  %int_f64 = arith.sitofp %int_result#0 : i64 to f64

  // Check if we have decimal point
  %has_decimal = arith.cmpi slt, %int_result#1, %len : i64

  // Parse fractional part if present
  %frac_start = arith.addi %int_result#1, %c1_i64 : i64

  %frac_result:3 = scf.while (%frac = %c0_f64, %div = %c1_f64, %pos = %frac_start) : (f64, f64, i64) -> (f64, f64, i64) {
    %in_bounds = arith.cmpi slt, %pos, %len : i64
    %continue = arith.andi %has_decimal, %in_bounds : i1
    scf.condition(%continue) %frac, %div, %pos : f64, f64, i64
  } do {
  ^bb0(%frac_arg: f64, %div_arg: f64, %pos_arg: i64):
    %char_ptr = llvm.getelementptr %ptr[%pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char = llvm.load %char_ptr : !llvm.ptr -> i8
    %char_i64 = arith.extui %char : i8 to i64
    %digit_i64 = arith.subi %char_i64, %c48 : i64
    %digit_f64 = arith.sitofp %digit_i64 : i64 to f64
    %new_div = arith.mulf %div_arg, %c10_f64 : f64
    %scaled_digit = arith.divf %digit_f64, %new_div : f64
    %new_frac = arith.addf %frac_arg, %scaled_digit : f64
    %new_pos = arith.addi %pos_arg, %c1_i64 : i64
    scf.yield %new_frac, %new_div, %new_pos : f64, f64, i64
  }

  // Combine integer and fractional parts
  %combined = arith.addf %int_f64, %frac_result#0 : f64

  // Apply sign
  %negated = arith.negf %combined : f64
  %final = arith.select %is_neg, %negated, %combined : f64

  return %final : f64
}
"""

/// Check if string contains a character
/// Input: fat string, i8 char
/// Output: i1 (bool)
let stringContainsCharBody = """
func.func @fidelity_string_contains_char(%str: !llvm.struct<(ptr, i64)>, %target: i8) -> i1 {
  %ptr = llvm.extractvalue %str[0] : !llvm.struct<(ptr, i64)>
  %len = llvm.extractvalue %str[1] : !llvm.struct<(ptr, i64)>

  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %false = arith.constant false

  %result:2 = scf.while (%found = %false, %pos = %c0) : (i1, i64) -> (i1, i64) {
    %in_bounds = arith.cmpi slt, %pos, %len : i64
    %not_found = arith.cmpi eq, %found, %false : i1
    %continue = arith.andi %in_bounds, %not_found : i1
    scf.condition(%continue) %found, %pos : i1, i64
  } do {
  ^bb0(%found_arg: i1, %pos_arg: i64):
    %char_ptr = llvm.getelementptr %ptr[%pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char = llvm.load %char_ptr : !llvm.ptr -> i8
    %is_match = arith.cmpi eq, %char, %target : i8
    %new_pos = arith.addi %pos_arg, %c1 : i64
    scf.yield %is_match, %new_pos : i1, i64
  }

  return %result#0 : i1
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// Helper Registration Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Register the Parse.int helper and return updated zipper
let ensureParseIntHelper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody ParseIntHelper parseIntBody zipper

/// Register the Parse.float helper and return updated zipper
let ensureParseFloatHelper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody ParseFloatHelper parseFloatBody zipper

/// Register the String.containsChar helper and return updated zipper
let ensureStringContainsCharHelper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody StringContainsCharHelper stringContainsCharBody zipper

// ═══════════════════════════════════════════════════════════════════════════
// Call Emission Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a call to fidelity_parse_int and return the result SSA
let emitParseIntCall (strSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // Register the helper if needed
    let zipper1 = ensureParseIntHelper zipper

    // Emit the call
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s) : (!llvm.struct<(ptr, i64)>) -> i64" resultSSA ParseIntHelper strSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I64) zipper2
    resultSSA, zipper3

/// Emit a call to fidelity_parse_float and return the result SSA
let emitParseFloatCall (strSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // Register the helper if needed
    let zipper1 = ensureParseFloatHelper zipper

    // Emit the call
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s) : (!llvm.struct<(ptr, i64)>) -> f64" resultSSA ParseFloatHelper strSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Float Alex.CodeGeneration.MLIRTypes.F64) zipper2
    resultSSA, zipper3

/// Emit a call to fidelity_string_contains_char and return the result SSA
let emitStringContainsCharCall (strSSA: string) (charSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // Register the helper if needed
    let zipper1 = ensureStringContainsCharHelper zipper

    // Emit the call
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s, %s) : (!llvm.struct<(ptr, i64)>, i8) -> i1" resultSSA StringContainsCharHelper strSSA charSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I1) zipper2
    resultSSA, zipper3

// ═══════════════════════════════════════════════════════════════════════════
// CRYPTO HELPERS - SHA-1 and Base64
// ═══════════════════════════════════════════════════════════════════════════

[<Literal>]
let Base64EncodeHelper = "fidelity_base64_encode"

[<Literal>]
let Base64DecodeHelper = "fidelity_base64_decode"

[<Literal>]
let Sha1Helper = "fidelity_sha1"

/// Base64 encoding lookup table (64 chars: A-Za-z0-9+/)
/// Returns encoded fat string (!llvm.struct<(ptr, i64)>)
let base64EncodeBody = """
func.func @fidelity_base64_encode(%input: !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)> {
  %input_ptr = llvm.extractvalue %input[0] : !llvm.struct<(ptr, i64)>
  %input_len = llvm.extractvalue %input[1] : !llvm.struct<(ptr, i64)>

  // Constants
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %c4 = arith.constant 4 : i64
  %c6 = arith.constant 6 : i64
  %c8 = arith.constant 8 : i64
  %c16 = arith.constant 16 : i64
  %c61_i8 = arith.constant 61 : i8
  %c63_i32 = arith.constant 63 : i32

  // Output length: ceil(input_len / 3) * 4
  %input_plus_2 = arith.addi %input_len, %c2 : i64
  %groups = arith.divui %input_plus_2, %c3 : i64
  %output_len = arith.muli %groups, %c4 : i64

  // Allocate output buffer
  %output_ptr = llvm.alloca %output_len x i8 : (i64) -> !llvm.ptr

  // Base64 alphabet as constants (A-Z=0-25, a-z=26-51, 0-9=52-61, +=62, /=63)
  // We'll compute char from index: 0-25 -> 'A'+i, 26-51 -> 'a'+(i-26), etc.

  // Process 3-byte groups
  %final:2 = scf.while (%in_pos = %c0, %out_pos = %c0) : (i64, i64) -> (i64, i64) {
    %has_more = arith.cmpi slt, %in_pos, %input_len : i64
    scf.condition(%has_more) %in_pos, %out_pos : i64, i64
  } do {
  ^bb0(%in_pos_arg: i64, %out_pos_arg: i64):
    // Read up to 3 bytes
    %b0_ptr = llvm.getelementptr %input_ptr[%in_pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %b0 = llvm.load %b0_ptr : !llvm.ptr -> i8
    %b0_i32 = arith.extui %b0 : i8 to i32

    %pos1 = arith.addi %in_pos_arg, %c1 : i64
    %has_b1 = arith.cmpi slt, %pos1, %input_len : i64
    %b1_ptr = llvm.getelementptr %input_ptr[%pos1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %b1_raw = llvm.load %b1_ptr : !llvm.ptr -> i8
    %b1_raw_i32 = arith.extui %b1_raw : i8 to i32
    %zero_i32 = arith.constant 0 : i32
    %b1_i32 = arith.select %has_b1, %b1_raw_i32, %zero_i32 : i32

    %pos2 = arith.addi %in_pos_arg, %c2 : i64
    %has_b2 = arith.cmpi slt, %pos2, %input_len : i64
    %b2_ptr = llvm.getelementptr %input_ptr[%pos2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %b2_raw = llvm.load %b2_ptr : !llvm.ptr -> i8
    %b2_raw_i32 = arith.extui %b2_raw : i8 to i32
    %b2_i32 = arith.select %has_b2, %b2_raw_i32, %zero_i32 : i32

    // Combine into 24-bit value: (b0 << 16) | (b1 << 8) | b2
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %b0_shifted = arith.shli %b0_i32, %c16_i32 : i32
    %b1_shifted = arith.shli %b1_i32, %c8_i32 : i32
    %combined_01 = arith.ori %b0_shifted, %b1_shifted : i32
    %combined = arith.ori %combined_01, %b2_i32 : i32

    // Extract 4 6-bit indices
    %c18_i32 = arith.constant 18 : i32
    %c12_i32 = arith.constant 12 : i32
    %c6_i32 = arith.constant 6 : i32
    %idx0 = arith.shrui %combined, %c18_i32 : i32
    %idx0_masked = arith.andi %idx0, %c63_i32 : i32
    %idx1 = arith.shrui %combined, %c12_i32 : i32
    %idx1_masked = arith.andi %idx1, %c63_i32 : i32
    %idx2 = arith.shrui %combined, %c6_i32 : i32
    %idx2_masked = arith.andi %idx2, %c63_i32 : i32
    %idx3_masked = arith.andi %combined, %c63_i32 : i32

    // Convert indices to Base64 characters
    // 0-25: 'A' (65) + idx
    // 26-51: 'a' (97) + (idx - 26)
    // 52-61: '0' (48) + (idx - 52)
    // 62: '+' (43)
    // 63: '/' (47)
    %c26_i32 = arith.constant 26 : i32
    %c52_i32 = arith.constant 52 : i32
    %c62_i32 = arith.constant 62 : i32
    %c65_i32 = arith.constant 65 : i32
    %c71_i32 = arith.constant 71 : i32
    %c_4_i32 = arith.constant -4 : i32
    %c43_i32 = arith.constant 43 : i32
    %c47_i32 = arith.constant 47 : i32

    // Char from index helper inline: compute for each index
    // idx < 26 ? 'A' + idx : idx < 52 ? 'a' + idx - 26 : idx < 62 ? '0' + idx - 52 : idx == 62 ? '+' : '/'
    %is_upper0 = arith.cmpi slt, %idx0_masked, %c26_i32 : i32
    %upper_char0 = arith.addi %idx0_masked, %c65_i32 : i32
    %is_lower0 = arith.cmpi slt, %idx0_masked, %c52_i32 : i32
    %lower_char0 = arith.addi %idx0_masked, %c71_i32 : i32
    %is_digit0 = arith.cmpi slt, %idx0_masked, %c62_i32 : i32
    %digit_char0 = arith.addi %idx0_masked, %c_4_i32 : i32
    %is_plus0 = arith.cmpi eq, %idx0_masked, %c62_i32 : i32
    %special0 = arith.select %is_plus0, %c43_i32, %c47_i32 : i32
    %char0_t1 = arith.select %is_digit0, %digit_char0, %special0 : i32
    %char0_t2 = arith.select %is_lower0, %lower_char0, %char0_t1 : i32
    %char0_i32 = arith.select %is_upper0, %upper_char0, %char0_t2 : i32
    %char0 = arith.trunci %char0_i32 : i32 to i8

    %is_upper1 = arith.cmpi slt, %idx1_masked, %c26_i32 : i32
    %upper_char1 = arith.addi %idx1_masked, %c65_i32 : i32
    %is_lower1 = arith.cmpi slt, %idx1_masked, %c52_i32 : i32
    %lower_char1 = arith.addi %idx1_masked, %c71_i32 : i32
    %is_digit1 = arith.cmpi slt, %idx1_masked, %c62_i32 : i32
    %digit_char1 = arith.addi %idx1_masked, %c_4_i32 : i32
    %is_plus1 = arith.cmpi eq, %idx1_masked, %c62_i32 : i32
    %special1 = arith.select %is_plus1, %c43_i32, %c47_i32 : i32
    %char1_t1 = arith.select %is_digit1, %digit_char1, %special1 : i32
    %char1_t2 = arith.select %is_lower1, %lower_char1, %char1_t1 : i32
    %char1_i32 = arith.select %is_upper1, %upper_char1, %char1_t2 : i32
    %char1 = arith.trunci %char1_i32 : i32 to i8

    %is_upper2 = arith.cmpi slt, %idx2_masked, %c26_i32 : i32
    %upper_char2 = arith.addi %idx2_masked, %c65_i32 : i32
    %is_lower2 = arith.cmpi slt, %idx2_masked, %c52_i32 : i32
    %lower_char2 = arith.addi %idx2_masked, %c71_i32 : i32
    %is_digit2 = arith.cmpi slt, %idx2_masked, %c62_i32 : i32
    %digit_char2 = arith.addi %idx2_masked, %c_4_i32 : i32
    %is_plus2 = arith.cmpi eq, %idx2_masked, %c62_i32 : i32
    %special2 = arith.select %is_plus2, %c43_i32, %c47_i32 : i32
    %char2_t1 = arith.select %is_digit2, %digit_char2, %special2 : i32
    %char2_t2 = arith.select %is_lower2, %lower_char2, %char2_t1 : i32
    %char2_i32 = arith.select %is_upper2, %upper_char2, %char2_t2 : i32
    %char2_pre = arith.trunci %char2_i32 : i32 to i8
    %char2 = arith.select %has_b1, %char2_pre, %c61_i8 : i8

    %is_upper3 = arith.cmpi slt, %idx3_masked, %c26_i32 : i32
    %upper_char3 = arith.addi %idx3_masked, %c65_i32 : i32
    %is_lower3 = arith.cmpi slt, %idx3_masked, %c52_i32 : i32
    %lower_char3 = arith.addi %idx3_masked, %c71_i32 : i32
    %is_digit3 = arith.cmpi slt, %idx3_masked, %c62_i32 : i32
    %digit_char3 = arith.addi %idx3_masked, %c_4_i32 : i32
    %is_plus3 = arith.cmpi eq, %idx3_masked, %c62_i32 : i32
    %special3 = arith.select %is_plus3, %c43_i32, %c47_i32 : i32
    %char3_t1 = arith.select %is_digit3, %digit_char3, %special3 : i32
    %char3_t2 = arith.select %is_lower3, %lower_char3, %char3_t1 : i32
    %char3_i32 = arith.select %is_upper3, %upper_char3, %char3_t2 : i32
    %char3_pre = arith.trunci %char3_i32 : i32 to i8
    %char3 = arith.select %has_b2, %char3_pre, %c61_i8 : i8

    // Store 4 output characters
    %out0_ptr = llvm.getelementptr %output_ptr[%out_pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %char0, %out0_ptr : i8, !llvm.ptr
    %out_pos_1 = arith.addi %out_pos_arg, %c1 : i64
    %out1_ptr = llvm.getelementptr %output_ptr[%out_pos_1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %char1, %out1_ptr : i8, !llvm.ptr
    %out_pos_2 = arith.addi %out_pos_arg, %c2 : i64
    %out2_ptr = llvm.getelementptr %output_ptr[%out_pos_2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %char2, %out2_ptr : i8, !llvm.ptr
    %out_pos_3 = arith.addi %out_pos_arg, %c3 : i64
    %out3_ptr = llvm.getelementptr %output_ptr[%out_pos_3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %char3, %out3_ptr : i8, !llvm.ptr

    %new_in_pos = arith.addi %in_pos_arg, %c3 : i64
    %new_out_pos = arith.addi %out_pos_arg, %c4 : i64
    scf.yield %new_in_pos, %new_out_pos : i64, i64
  }

  // Build result fat string
  %result_undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %result_with_ptr = llvm.insertvalue %output_ptr, %result_undef[0] : !llvm.struct<(ptr, i64)>
  %result = llvm.insertvalue %output_len, %result_with_ptr[1] : !llvm.struct<(ptr, i64)>
  return %result : !llvm.struct<(ptr, i64)>
}
"""

/// Base64 decoding
/// Returns decoded fat array (!llvm.struct<(ptr, i64)>)
let base64DecodeBody = """
func.func @fidelity_base64_decode(%input: !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)> {
  %input_ptr = llvm.extractvalue %input[0] : !llvm.struct<(ptr, i64)>
  %input_len = llvm.extractvalue %input[1] : !llvm.struct<(ptr, i64)>

  // Constants
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %c4 = arith.constant 4 : i64

  // Count padding ('=' chars at end)
  %len_minus_1 = arith.subi %input_len, %c1 : i64
  %len_minus_2 = arith.subi %input_len, %c2 : i64
  %last_ptr = llvm.getelementptr %input_ptr[%len_minus_1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %last_char = llvm.load %last_ptr : !llvm.ptr -> i8
  %c61_i8 = arith.constant 61 : i8
  %last_is_pad = arith.cmpi eq, %last_char, %c61_i8 : i8
  %second_last_ptr = llvm.getelementptr %input_ptr[%len_minus_2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %second_last_char = llvm.load %second_last_ptr : !llvm.ptr -> i8
  %second_last_is_pad = arith.cmpi eq, %second_last_char, %c61_i8 : i8
  %one_pad = arith.select %last_is_pad, %c1, %c0 : i64
  %two_pad_cond = arith.andi %last_is_pad, %second_last_is_pad : i1
  %padding = arith.select %two_pad_cond, %c2, %one_pad : i64

  // Output length: (input_len / 4) * 3 - padding
  %groups = arith.divui %input_len, %c4 : i64
  %base_output = arith.muli %groups, %c3 : i64
  %output_len = arith.subi %base_output, %padding : i64

  // Allocate output buffer
  %output_ptr = llvm.alloca %output_len x i8 : (i64) -> !llvm.ptr

  // Process 4-char groups
  %final:2 = scf.while (%in_pos = %c0, %out_pos = %c0) : (i64, i64) -> (i64, i64) {
    %has_more = arith.cmpi slt, %in_pos, %input_len : i64
    scf.condition(%has_more) %in_pos, %out_pos : i64, i64
  } do {
  ^bb0(%in_pos_arg: i64, %out_pos_arg: i64):
    // Read 4 Base64 characters
    %c0_ptr = llvm.getelementptr %input_ptr[%in_pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char0 = llvm.load %c0_ptr : !llvm.ptr -> i8
    %in_pos_1 = arith.addi %in_pos_arg, %c1 : i64
    %c1_ptr = llvm.getelementptr %input_ptr[%in_pos_1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char1 = llvm.load %c1_ptr : !llvm.ptr -> i8
    %in_pos_2 = arith.addi %in_pos_arg, %c2 : i64
    %c2_ptr = llvm.getelementptr %input_ptr[%in_pos_2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char2 = llvm.load %c2_ptr : !llvm.ptr -> i8
    %in_pos_3 = arith.addi %in_pos_arg, %c3 : i64
    %c3_ptr = llvm.getelementptr %input_ptr[%in_pos_3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char3 = llvm.load %c3_ptr : !llvm.ptr -> i8

    // Decode each character to 6-bit value
    // 'A'-'Z' (65-90) -> 0-25
    // 'a'-'z' (97-122) -> 26-51
    // '0'-'9' (48-57) -> 52-61
    // '+' (43) -> 62
    // '/' (47) -> 63
    // '=' (61) -> 0 (padding)
    %c65_i8 = arith.constant 65 : i8
    %c97_i8 = arith.constant 97 : i8
    %c48_i8 = arith.constant 48 : i8
    %c43_i8 = arith.constant 43 : i8
    %c47_i8 = arith.constant 47 : i8
    %c26_i8 = arith.constant 26 : i8
    %c52_i8 = arith.constant 52 : i8
    %c62_i8 = arith.constant 62 : i8
    %c63_i8 = arith.constant 63 : i8
    %c0_i8 = arith.constant 0 : i8
    %c90_i8 = arith.constant 90 : i8
    %c122_i8 = arith.constant 122 : i8
    %c57_i8 = arith.constant 57 : i8

    // Decode char0
    %is_upper0 = arith.cmpi sle, %char0, %c90_i8 : i8
    %val_upper0 = arith.subi %char0, %c65_i8 : i8
    %is_lower0 = arith.cmpi sle, %char0, %c122_i8 : i8
    %val_lower0_raw = arith.subi %char0, %c97_i8 : i8
    %val_lower0 = arith.addi %val_lower0_raw, %c26_i8 : i8
    %is_digit0 = arith.cmpi sle, %char0, %c57_i8 : i8
    %val_digit0_raw = arith.subi %char0, %c48_i8 : i8
    %val_digit0 = arith.addi %val_digit0_raw, %c52_i8 : i8
    %is_plus0 = arith.cmpi eq, %char0, %c43_i8 : i8
    %is_slash0 = arith.cmpi eq, %char0, %c47_i8 : i8
    %val_slash0 = arith.select %is_slash0, %c63_i8, %c0_i8 : i8
    %val_plus0 = arith.select %is_plus0, %c62_i8, %val_slash0 : i8
    %val_digit_or0 = arith.select %is_digit0, %val_digit0, %val_plus0 : i8
    %val_lower_or0 = arith.select %is_lower0, %val_lower0, %val_digit_or0 : i8
    %idx0 = arith.select %is_upper0, %val_upper0, %val_lower_or0 : i8
    %idx0_i32 = arith.extui %idx0 : i8 to i32

    // Decode char1
    %is_upper1 = arith.cmpi sle, %char1, %c90_i8 : i8
    %val_upper1 = arith.subi %char1, %c65_i8 : i8
    %is_lower1 = arith.cmpi sle, %char1, %c122_i8 : i8
    %val_lower1_raw = arith.subi %char1, %c97_i8 : i8
    %val_lower1 = arith.addi %val_lower1_raw, %c26_i8 : i8
    %is_digit1 = arith.cmpi sle, %char1, %c57_i8 : i8
    %val_digit1_raw = arith.subi %char1, %c48_i8 : i8
    %val_digit1 = arith.addi %val_digit1_raw, %c52_i8 : i8
    %is_plus1 = arith.cmpi eq, %char1, %c43_i8 : i8
    %is_slash1 = arith.cmpi eq, %char1, %c47_i8 : i8
    %val_slash1 = arith.select %is_slash1, %c63_i8, %c0_i8 : i8
    %val_plus1 = arith.select %is_plus1, %c62_i8, %val_slash1 : i8
    %val_digit_or1 = arith.select %is_digit1, %val_digit1, %val_plus1 : i8
    %val_lower_or1 = arith.select %is_lower1, %val_lower1, %val_digit_or1 : i8
    %idx1 = arith.select %is_upper1, %val_upper1, %val_lower_or1 : i8
    %idx1_i32 = arith.extui %idx1 : i8 to i32

    // Decode char2 (may be padding)
    %is_pad2 = arith.cmpi eq, %char2, %c61_i8 : i8
    %is_upper2 = arith.cmpi sle, %char2, %c90_i8 : i8
    %val_upper2 = arith.subi %char2, %c65_i8 : i8
    %is_lower2 = arith.cmpi sle, %char2, %c122_i8 : i8
    %val_lower2_raw = arith.subi %char2, %c97_i8 : i8
    %val_lower2 = arith.addi %val_lower2_raw, %c26_i8 : i8
    %is_digit2 = arith.cmpi sle, %char2, %c57_i8 : i8
    %val_digit2_raw = arith.subi %char2, %c48_i8 : i8
    %val_digit2 = arith.addi %val_digit2_raw, %c52_i8 : i8
    %is_plus2 = arith.cmpi eq, %char2, %c43_i8 : i8
    %is_slash2 = arith.cmpi eq, %char2, %c47_i8 : i8
    %val_slash2 = arith.select %is_slash2, %c63_i8, %c0_i8 : i8
    %val_plus2 = arith.select %is_plus2, %c62_i8, %val_slash2 : i8
    %val_digit_or2 = arith.select %is_digit2, %val_digit2, %val_plus2 : i8
    %val_lower_or2 = arith.select %is_lower2, %val_lower2, %val_digit_or2 : i8
    %val_decoded2 = arith.select %is_upper2, %val_upper2, %val_lower_or2 : i8
    %idx2 = arith.select %is_pad2, %c0_i8, %val_decoded2 : i8
    %idx2_i32 = arith.extui %idx2 : i8 to i32

    // Decode char3 (may be padding)
    %is_pad3 = arith.cmpi eq, %char3, %c61_i8 : i8
    %is_upper3 = arith.cmpi sle, %char3, %c90_i8 : i8
    %val_upper3 = arith.subi %char3, %c65_i8 : i8
    %is_lower3 = arith.cmpi sle, %char3, %c122_i8 : i8
    %val_lower3_raw = arith.subi %char3, %c97_i8 : i8
    %val_lower3 = arith.addi %val_lower3_raw, %c26_i8 : i8
    %is_digit3 = arith.cmpi sle, %char3, %c57_i8 : i8
    %val_digit3_raw = arith.subi %char3, %c48_i8 : i8
    %val_digit3 = arith.addi %val_digit3_raw, %c52_i8 : i8
    %is_plus3 = arith.cmpi eq, %char3, %c43_i8 : i8
    %is_slash3 = arith.cmpi eq, %char3, %c47_i8 : i8
    %val_slash3 = arith.select %is_slash3, %c63_i8, %c0_i8 : i8
    %val_plus3 = arith.select %is_plus3, %c62_i8, %val_slash3 : i8
    %val_digit_or3 = arith.select %is_digit3, %val_digit3, %val_plus3 : i8
    %val_lower_or3 = arith.select %is_lower3, %val_lower3, %val_digit_or3 : i8
    %val_decoded3 = arith.select %is_upper3, %val_upper3, %val_lower_or3 : i8
    %idx3 = arith.select %is_pad3, %c0_i8, %val_decoded3 : i8
    %idx3_i32 = arith.extui %idx3 : i8 to i32

    // Combine into 24-bit value: (idx0 << 18) | (idx1 << 12) | (idx2 << 6) | idx3
    %c18_i32 = arith.constant 18 : i32
    %c12_i32 = arith.constant 12 : i32
    %c6_i32 = arith.constant 6 : i32
    %idx0_shifted = arith.shli %idx0_i32, %c18_i32 : i32
    %idx1_shifted = arith.shli %idx1_i32, %c12_i32 : i32
    %idx2_shifted = arith.shli %idx2_i32, %c6_i32 : i32
    %combined_01 = arith.ori %idx0_shifted, %idx1_shifted : i32
    %combined_012 = arith.ori %combined_01, %idx2_shifted : i32
    %combined = arith.ori %combined_012, %idx3_i32 : i32

    // Extract 3 bytes
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    %b0_i32 = arith.shrui %combined, %c16_i32 : i32
    %b0_masked = arith.andi %b0_i32, %c255_i32 : i32
    %b0 = arith.trunci %b0_masked : i32 to i8
    %b1_i32 = arith.shrui %combined, %c8_i32 : i32
    %b1_masked = arith.andi %b1_i32, %c255_i32 : i32
    %b1 = arith.trunci %b1_masked : i32 to i8
    %b2_masked = arith.andi %combined, %c255_i32 : i32
    %b2 = arith.trunci %b2_masked : i32 to i8

    // Store bytes (respecting padding)
    %out0_ptr = llvm.getelementptr %output_ptr[%out_pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %b0, %out0_ptr : i8, !llvm.ptr

    %out_pos_1 = arith.addi %out_pos_arg, %c1 : i64
    %write_b1 = arith.cmpi slt, %out_pos_1, %output_len : i64
    scf.if %write_b1 {
      %out1_ptr = llvm.getelementptr %output_ptr[%out_pos_1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      llvm.store %b1, %out1_ptr : i8, !llvm.ptr
    }

    %out_pos_2 = arith.addi %out_pos_arg, %c2 : i64
    %write_b2 = arith.cmpi slt, %out_pos_2, %output_len : i64
    scf.if %write_b2 {
      %out2_ptr = llvm.getelementptr %output_ptr[%out_pos_2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      llvm.store %b2, %out2_ptr : i8, !llvm.ptr
    }

    %new_in_pos = arith.addi %in_pos_arg, %c4 : i64
    %new_out_pos = arith.addi %out_pos_arg, %c3 : i64
    scf.yield %new_in_pos, %new_out_pos : i64, i64
  }

  // Build result fat array
  %result_undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %result_with_ptr = llvm.insertvalue %output_ptr, %result_undef[0] : !llvm.struct<(ptr, i64)>
  %result = llvm.insertvalue %output_len, %result_with_ptr[1] : !llvm.struct<(ptr, i64)>
  return %result : !llvm.struct<(ptr, i64)>
}
"""

/// SHA-1 hash function (FIPS 180-4)
/// Input: fat array (!llvm.struct<(ptr, i64)>)
/// Output: fat array with 20-byte hash (!llvm.struct<(ptr, i64)>)
let sha1Body = """
func.func @fidelity_sha1(%input: !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)> {
  %input_ptr = llvm.extractvalue %input[0] : !llvm.struct<(ptr, i64)>
  %input_len = llvm.extractvalue %input[1] : !llvm.struct<(ptr, i64)>

  // Initialize hash state (H0-H4) per FIPS 180-4
  %h0_init = arith.constant 0x67452301 : i32
  %h1_init = arith.constant 0xEFCDAB89 : i32
  %h2_init = arith.constant 0x98BADCFE : i32
  %h3_init = arith.constant 0x10325476 : i32
  %h4_init = arith.constant 0xC3D2E1F0 : i32

  // Constants
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %c8 = arith.constant 8 : i64
  %c20 = arith.constant 20 : i64
  %c55 = arith.constant 55 : i64
  %c56 = arith.constant 56 : i64
  %c64 = arith.constant 64 : i64
  %c80 = arith.constant 80 : i64
  %c0_i8 = arith.constant 0 : i8
  %c128_i8 = arith.constant 128 : i8

  // SHA-1 round constants
  %k0 = arith.constant 0x5A827999 : i32
  %k1 = arith.constant 0x6ED9EBA1 : i32
  %k2 = arith.constant 0x8F1BBCDC : i32
  %k3 = arith.constant 0xCA62C1D6 : i32

  // Calculate padded message length
  // Padding: 1 bit, then zeros, then 64-bit length
  // Total must be multiple of 512 bits (64 bytes)
  %len_mod_64 = arith.remui %input_len, %c64 : i64
  %needs_extra_block = arith.cmpi sge, %len_mod_64, %c56 : i64
  %base_padding = arith.subi %c64, %len_mod_64 : i64
  %extra_block = arith.select %needs_extra_block, %c64, %c0 : i64
  %padding_len = arith.addi %base_padding, %extra_block : i64
  %padded_len = arith.addi %input_len, %padding_len : i64

  // Allocate padded message buffer
  %padded_ptr = llvm.alloca %padded_len x i8 : (i64) -> !llvm.ptr

  // Copy input to padded buffer
  "llvm.intr.memcpy"(%padded_ptr, %input_ptr, %input_len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()

  // Add 0x80 byte after input
  %pad_start_ptr = llvm.getelementptr %padded_ptr[%input_len] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %c128_i8, %pad_start_ptr : i8, !llvm.ptr

  // Zero-fill padding (except last 8 bytes for length)
  %zero_start = arith.addi %input_len, %c1 : i64
  %zero_end = arith.subi %padded_len, %c8 : i64
  scf.for %i = %zero_start to %zero_end step %c1 {
    %zero_ptr = llvm.getelementptr %padded_ptr[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %c0_i8, %zero_ptr : i8, !llvm.ptr
  }

  // Append length in bits (big-endian, 64-bit)
  %len_bits = arith.muli %input_len, %c8 : i64
  %len_pos = arith.subi %padded_len, %c8 : i64
  scf.for %i = %c0 to %c8 step %c1 {
    %shift = arith.subi %c8, %i : i64
    %shift_minus_1 = arith.subi %shift, %c1 : i64
    %shift_bits_64 = arith.muli %shift_minus_1, %c8 : i64
    %shift_bits = arith.trunci %shift_bits_64 : i64 to i32
    %len_bits_32 = arith.trunci %len_bits : i64 to i32
    %byte_val_32 = arith.shrui %len_bits_32, %shift_bits : i32
    %byte_val = arith.trunci %byte_val_32 : i32 to i8
    %byte_pos = arith.addi %len_pos, %i : i64
    %byte_ptr = llvm.getelementptr %padded_ptr[%byte_pos] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %byte_val, %byte_ptr : i8, !llvm.ptr
  }

  // Allocate message schedule (80 32-bit words)
  %w_ptr = llvm.alloca %c80 x i32 : (i64) -> !llvm.ptr

  // Process each 64-byte block
  %num_blocks = arith.divui %padded_len, %c64 : i64

  %final_h:5 = scf.for %block_idx = %c0 to %num_blocks step %c1
      iter_args(%h0 = %h0_init, %h1 = %h1_init, %h2 = %h2_init, %h3 = %h3_init, %h4 = %h4_init) -> (i32, i32, i32, i32, i32) {
    %block_offset = arith.muli %block_idx, %c64 : i64
    %block_ptr = llvm.getelementptr %padded_ptr[%block_offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8

    // Load 16 words from block (big-endian)
    scf.for %i = %c0 to %c64 step %c4 {
      %word_idx = arith.divui %i, %c4 : i64
      %b0_ptr = llvm.getelementptr %block_ptr[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %b0 = llvm.load %b0_ptr : !llvm.ptr -> i8
      %i1 = arith.addi %i, %c1 : i64
      %b1_ptr = llvm.getelementptr %block_ptr[%i1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %b1 = llvm.load %b1_ptr : !llvm.ptr -> i8
      %i2 = arith.addi %i, %c1 : i64
      %i2_2 = arith.addi %i2, %c1 : i64
      %b2_ptr = llvm.getelementptr %block_ptr[%i2_2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %b2 = llvm.load %b2_ptr : !llvm.ptr -> i8
      %i3 = arith.addi %i2_2, %c1 : i64
      %b3_ptr = llvm.getelementptr %block_ptr[%i3] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %b3 = llvm.load %b3_ptr : !llvm.ptr -> i8
      %b0_32 = arith.extui %b0 : i8 to i32
      %b1_32 = arith.extui %b1 : i8 to i32
      %b2_32 = arith.extui %b2 : i8 to i32
      %b3_32 = arith.extui %b3 : i8 to i32
      %c24_i32 = arith.constant 24 : i32
      %c16_i32 = arith.constant 16 : i32
      %c8_i32 = arith.constant 8 : i32
      %w0 = arith.shli %b0_32, %c24_i32 : i32
      %w1 = arith.shli %b1_32, %c16_i32 : i32
      %w2 = arith.shli %b2_32, %c8_i32 : i32
      %word_01 = arith.ori %w0, %w1 : i32
      %word_012 = arith.ori %word_01, %w2 : i32
      %word = arith.ori %word_012, %b3_32 : i32
      %w_word_ptr = llvm.getelementptr %w_ptr[%word_idx] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %word, %w_word_ptr : i32, !llvm.ptr
    }

    // Extend to 80 words
    %c16 = arith.constant 16 : i64
    scf.for %i = %c16 to %c80 step %c1 {
      %im3 = arith.subi %i, %c4 : i64
      %im3_1 = arith.addi %im3, %c1 : i64
      %im8 = arith.subi %i, %c8 : i64
      %im14 = arith.subi %im8, %c4 : i64
      %im14_2 = arith.subi %im14, %c1 : i64
      %im14_1 = arith.subi %im14_2, %c1 : i64
      %im16 = arith.subi %i, %c16 : i64
      %w_im3_ptr = llvm.getelementptr %w_ptr[%im3_1] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %w_im3 = llvm.load %w_im3_ptr : !llvm.ptr -> i32
      %w_im8_ptr = llvm.getelementptr %w_ptr[%im8] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %w_im8 = llvm.load %w_im8_ptr : !llvm.ptr -> i32
      %w_im14_ptr = llvm.getelementptr %w_ptr[%im14_1] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %w_im14 = llvm.load %w_im14_ptr : !llvm.ptr -> i32
      %w_im16_ptr = llvm.getelementptr %w_ptr[%im16] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %w_im16 = llvm.load %w_im16_ptr : !llvm.ptr -> i32
      %xor1 = arith.xori %w_im3, %w_im8 : i32
      %xor2 = arith.xori %xor1, %w_im14 : i32
      %xor3 = arith.xori %xor2, %w_im16 : i32
      // Left rotate by 1
      %c1_i32 = arith.constant 1 : i32
      %c31_i32 = arith.constant 31 : i32
      %rotl = arith.shli %xor3, %c1_i32 : i32
      %rotr = arith.shrui %xor3, %c31_i32 : i32
      %w_i = arith.ori %rotl, %rotr : i32
      %w_i_ptr = llvm.getelementptr %w_ptr[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %w_i, %w_i_ptr : i32, !llvm.ptr
    }

    // Main loop (80 rounds)
    %round_result:5 = scf.for %round = %c0 to %c80 step %c1
        iter_args(%a = %h0, %b = %h1, %c = %h2, %d = %h3, %e = %h4) -> (i32, i32, i32, i32, i32) {
      // Determine f and k based on round
      %c20_r = arith.constant 20 : i64
      %c40_r = arith.constant 40 : i64
      %c60_r = arith.constant 60 : i64

      %is_r0 = arith.cmpi slt, %round, %c20_r : i64
      %is_r1 = arith.cmpi slt, %round, %c40_r : i64
      %is_r2 = arith.cmpi slt, %round, %c60_r : i64

      // f = (b & c) | (~b & d) for rounds 0-19
      // f = b ^ c ^ d for rounds 20-39, 60-79
      // f = (b & c) | (b & d) | (c & d) for rounds 40-59
      %b_and_c = arith.andi %b, %c : i32
      %not_b = arith.constant -1 : i32
      %b_inv = arith.xori %b, %not_b : i32
      %not_b_and_d = arith.andi %b_inv, %d : i32
      %f0 = arith.ori %b_and_c, %not_b_and_d : i32

      %f1 = arith.xori %b, %c : i32
      %f1_full = arith.xori %f1, %d : i32

      %b_and_d = arith.andi %b, %d : i32
      %c_and_d = arith.andi %c, %d : i32
      %f2_part = arith.ori %b_and_c, %b_and_d : i32
      %f2 = arith.ori %f2_part, %c_and_d : i32

      %f_r1 = arith.select %is_r2, %f2, %f1_full : i32
      %f_r0 = arith.select %is_r1, %f_r1, %f1_full : i32
      %f = arith.select %is_r0, %f0, %f_r0 : i32

      %k_r1 = arith.select %is_r2, %k2, %k3 : i32
      %k_r0 = arith.select %is_r1, %k_r1, %k1 : i32
      %k = arith.select %is_r0, %k0, %k_r0 : i32

      // temp = (a <<< 5) + f + e + k + w[i]
      %c5_i32 = arith.constant 5 : i32
      %c27_i32 = arith.constant 27 : i32
      %a_rotl = arith.shli %a, %c5_i32 : i32
      %a_rotr = arith.shrui %a, %c27_i32 : i32
      %a_rot = arith.ori %a_rotl, %a_rotr : i32

      %w_round_ptr = llvm.getelementptr %w_ptr[%round] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %w_round = llvm.load %w_round_ptr : !llvm.ptr -> i32

      %temp1 = arith.addi %a_rot, %f : i32
      %temp2 = arith.addi %temp1, %e : i32
      %temp3 = arith.addi %temp2, %k : i32
      %temp = arith.addi %temp3, %w_round : i32

      // e = d, d = c, c = b <<< 30, b = a, a = temp
      %c30_i32 = arith.constant 30 : i32
      %c2_i32 = arith.constant 2 : i32
      %b_rotl = arith.shli %b, %c30_i32 : i32
      %b_rotr = arith.shrui %b, %c2_i32 : i32
      %new_c = arith.ori %b_rotl, %b_rotr : i32

      scf.yield %temp, %a, %new_c, %c, %d : i32, i32, i32, i32, i32
    }

    // Add round result to hash state
    %new_h0 = arith.addi %h0, %round_result#0 : i32
    %new_h1 = arith.addi %h1, %round_result#1 : i32
    %new_h2 = arith.addi %h2, %round_result#2 : i32
    %new_h3 = arith.addi %h3, %round_result#3 : i32
    %new_h4 = arith.addi %h4, %round_result#4 : i32

    scf.yield %new_h0, %new_h1, %new_h2, %new_h3, %new_h4 : i32, i32, i32, i32, i32
  }

  // Allocate 20-byte output buffer
  %output_ptr = llvm.alloca %c20 x i8 : (i64) -> !llvm.ptr

  // Write hash words to output (big-endian)
  %c24_i32_out = arith.constant 24 : i32
  %c16_i32_out = arith.constant 16 : i32
  %c8_i32_out = arith.constant 8 : i32

  // H0 bytes 0-3
  %h0_b0 = arith.shrui %final_h#0, %c24_i32_out : i32
  %h0_b0_i8 = arith.trunci %h0_b0 : i32 to i8
  %out0 = llvm.getelementptr %output_ptr[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h0_b0_i8, %out0 : i8, !llvm.ptr
  %h0_b1 = arith.shrui %final_h#0, %c16_i32_out : i32
  %h0_b1_i8 = arith.trunci %h0_b1 : i32 to i8
  %out1 = llvm.getelementptr %output_ptr[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h0_b1_i8, %out1 : i8, !llvm.ptr
  %h0_b2 = arith.shrui %final_h#0, %c8_i32_out : i32
  %h0_b2_i8 = arith.trunci %h0_b2 : i32 to i8
  %out2 = llvm.getelementptr %output_ptr[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h0_b2_i8, %out2 : i8, !llvm.ptr
  %h0_b3_i8 = arith.trunci %final_h#0 : i32 to i8
  %c3_out = arith.constant 3 : i64
  %out3 = llvm.getelementptr %output_ptr[%c3_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h0_b3_i8, %out3 : i8, !llvm.ptr

  // H1 bytes 4-7
  %h1_b0 = arith.shrui %final_h#1, %c24_i32_out : i32
  %h1_b0_i8 = arith.trunci %h1_b0 : i32 to i8
  %out4 = llvm.getelementptr %output_ptr[%c4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h1_b0_i8, %out4 : i8, !llvm.ptr
  %c5_out = arith.constant 5 : i64
  %h1_b1 = arith.shrui %final_h#1, %c16_i32_out : i32
  %h1_b1_i8 = arith.trunci %h1_b1 : i32 to i8
  %out5 = llvm.getelementptr %output_ptr[%c5_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h1_b1_i8, %out5 : i8, !llvm.ptr
  %c6_out = arith.constant 6 : i64
  %h1_b2 = arith.shrui %final_h#1, %c8_i32_out : i32
  %h1_b2_i8 = arith.trunci %h1_b2 : i32 to i8
  %out6 = llvm.getelementptr %output_ptr[%c6_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h1_b2_i8, %out6 : i8, !llvm.ptr
  %c7_out = arith.constant 7 : i64
  %h1_b3_i8 = arith.trunci %final_h#1 : i32 to i8
  %out7 = llvm.getelementptr %output_ptr[%c7_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h1_b3_i8, %out7 : i8, !llvm.ptr

  // H2 bytes 8-11
  %h2_b0 = arith.shrui %final_h#2, %c24_i32_out : i32
  %h2_b0_i8 = arith.trunci %h2_b0 : i32 to i8
  %out8 = llvm.getelementptr %output_ptr[%c8] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h2_b0_i8, %out8 : i8, !llvm.ptr
  %c9_out = arith.constant 9 : i64
  %h2_b1 = arith.shrui %final_h#2, %c16_i32_out : i32
  %h2_b1_i8 = arith.trunci %h2_b1 : i32 to i8
  %out9 = llvm.getelementptr %output_ptr[%c9_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h2_b1_i8, %out9 : i8, !llvm.ptr
  %c10_out = arith.constant 10 : i64
  %h2_b2 = arith.shrui %final_h#2, %c8_i32_out : i32
  %h2_b2_i8 = arith.trunci %h2_b2 : i32 to i8
  %out10 = llvm.getelementptr %output_ptr[%c10_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h2_b2_i8, %out10 : i8, !llvm.ptr
  %c11_out = arith.constant 11 : i64
  %h2_b3_i8 = arith.trunci %final_h#2 : i32 to i8
  %out11 = llvm.getelementptr %output_ptr[%c11_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h2_b3_i8, %out11 : i8, !llvm.ptr

  // H3 bytes 12-15
  %c12_out = arith.constant 12 : i64
  %h3_b0 = arith.shrui %final_h#3, %c24_i32_out : i32
  %h3_b0_i8 = arith.trunci %h3_b0 : i32 to i8
  %out12 = llvm.getelementptr %output_ptr[%c12_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h3_b0_i8, %out12 : i8, !llvm.ptr
  %c13_out = arith.constant 13 : i64
  %h3_b1 = arith.shrui %final_h#3, %c16_i32_out : i32
  %h3_b1_i8 = arith.trunci %h3_b1 : i32 to i8
  %out13 = llvm.getelementptr %output_ptr[%c13_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h3_b1_i8, %out13 : i8, !llvm.ptr
  %c14_out = arith.constant 14 : i64
  %h3_b2 = arith.shrui %final_h#3, %c8_i32_out : i32
  %h3_b2_i8 = arith.trunci %h3_b2 : i32 to i8
  %out14 = llvm.getelementptr %output_ptr[%c14_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h3_b2_i8, %out14 : i8, !llvm.ptr
  %c15_out = arith.constant 15 : i64
  %h3_b3_i8 = arith.trunci %final_h#3 : i32 to i8
  %out15 = llvm.getelementptr %output_ptr[%c15_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h3_b3_i8, %out15 : i8, !llvm.ptr

  // H4 bytes 16-19
  %c16_out = arith.constant 16 : i64
  %h4_b0 = arith.shrui %final_h#4, %c24_i32_out : i32
  %h4_b0_i8 = arith.trunci %h4_b0 : i32 to i8
  %out16 = llvm.getelementptr %output_ptr[%c16_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h4_b0_i8, %out16 : i8, !llvm.ptr
  %c17_out = arith.constant 17 : i64
  %h4_b1 = arith.shrui %final_h#4, %c16_i32_out : i32
  %h4_b1_i8 = arith.trunci %h4_b1 : i32 to i8
  %out17 = llvm.getelementptr %output_ptr[%c17_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h4_b1_i8, %out17 : i8, !llvm.ptr
  %c18_out = arith.constant 18 : i64
  %h4_b2 = arith.shrui %final_h#4, %c8_i32_out : i32
  %h4_b2_i8 = arith.trunci %h4_b2 : i32 to i8
  %out18 = llvm.getelementptr %output_ptr[%c18_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h4_b2_i8, %out18 : i8, !llvm.ptr
  %c19_out = arith.constant 19 : i64
  %h4_b3_i8 = arith.trunci %final_h#4 : i32 to i8
  %out19 = llvm.getelementptr %output_ptr[%c19_out] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  llvm.store %h4_b3_i8, %out19 : i8, !llvm.ptr

  // Build result fat array
  %result_undef = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %result_with_ptr = llvm.insertvalue %output_ptr, %result_undef[0] : !llvm.struct<(ptr, i64)>
  %result = llvm.insertvalue %c20, %result_with_ptr[1] : !llvm.struct<(ptr, i64)>
  return %result : !llvm.struct<(ptr, i64)>
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// Crypto Helper Registration
// ═══════════════════════════════════════════════════════════════════════════

let ensureBase64EncodeHelper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody Base64EncodeHelper base64EncodeBody zipper

let ensureBase64DecodeHelper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody Base64DecodeHelper base64DecodeBody zipper

let ensureSha1Helper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody Sha1Helper sha1Body zipper

// ═══════════════════════════════════════════════════════════════════════════
// Crypto Call Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a call to fidelity_base64_encode
let emitBase64EncodeCall (inputSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    let zipper1 = ensureBase64EncodeHelper zipper
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s) : (!llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)>" resultSSA Base64EncodeHelper inputSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Struct [Alex.CodeGeneration.MLIRTypes.Pointer; Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I64]) zipper2
    resultSSA, zipper3

/// Emit a call to fidelity_base64_decode
let emitBase64DecodeCall (inputSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    let zipper1 = ensureBase64DecodeHelper zipper
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s) : (!llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)>" resultSSA Base64DecodeHelper inputSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Struct [Alex.CodeGeneration.MLIRTypes.Pointer; Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I64]) zipper2
    resultSSA, zipper3

/// Emit a call to fidelity_sha1
let emitSha1Call (inputSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    let zipper1 = ensureSha1Helper zipper
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s) : (!llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)>" resultSSA Sha1Helper inputSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Struct [Alex.CodeGeneration.MLIRTypes.Pointer; Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I64]) zipper2
    resultSSA, zipper3
