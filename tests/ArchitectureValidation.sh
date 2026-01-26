#!/bin/bash
# ArchitectureValidation.sh - Validate architectural constraints
#
# Usage: ./tests/ArchitectureValidation.sh
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed

set +e  # Don't exit on errors - collect all violations

WITNESS_DIR="src/Alex/Witnesses"
MAX_WITNESS_SIZE=300
VIOLATIONS=0

echo "═══════════════════════════════════════════════════════════════════════════"
echo "Architecture Validation"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1: Witness Size Limit
# ═══════════════════════════════════════════════════════════════════════════

echo "━━━ CHECK 1: Witness Size Limit (max $MAX_WITNESS_SIZE lines) ━━━"
echo ""

# Check witnesses recursively (including subdirectories like ControlFlow/)
while IFS= read -r -d '' file; do
    # Get relative path from WITNESS_DIR
    relpath="${file#$WITNESS_DIR/}"
    lines=$(wc -l < "$file")

    if [ "$lines" -gt "$MAX_WITNESS_SIZE" ]; then
        echo "❌ FAIL: $relpath is $lines lines (exceeds $MAX_WITNESS_SIZE)"
        echo "   → Factor into smaller focused witnesses"
        ((VIOLATIONS++))
    else
        echo "✅ PASS: $relpath ($lines lines)"
    fi
done < <(find "$WITNESS_DIR" -name "*.fs" -type f -print0 | sort -z)

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2: No Transform Logic in Witnesses
# ═══════════════════════════════════════════════════════════════════════════

echo "━━━ CHECK 2: No Transform Logic in Witnesses ━━━"
echo ""

# Check for String operations that should be FNCS decomposition
if grep -r "witnessStringConcat\|witnessStringContains" "$WITNESS_DIR"/*.fs 2>/dev/null; then
    echo "❌ FAIL: Transform logic found in witnesses"
    echo "   → String.concat and String.contains must be FNCS decomposition"
    ((VIOLATIONS++))
else
    echo "✅ PASS: No String transform logic found"
fi

# Check for DateTime arithmetic that should be FNCS intrinsics
if grep -r "DateTime.hour\|DateTime.minute\|DateTime.second" "$WITNESS_DIR"/*.fs 2>/dev/null | grep -v "^\s*//" | grep -v "needs FNCS"; then
    echo "❌ FAIL: DateTime arithmetic found in witnesses"
    echo "   → DateTime accessors must be FNCS intrinsics"
    ((VIOLATIONS++))
else
    echo "✅ PASS: No DateTime arithmetic found"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3: MLIRTransfer.fs is Clean
# ═══════════════════════════════════════════════════════════════════════════

echo "━━━ CHECK 3: MLIRTransfer.fs Cleanliness ━━━"
echo ""

MLIR_TRANSFER="src/Alex/Traversal/MLIRTransfer.fs"

# Check line count (should be around 965 lines, give or take 50)
mlir_lines=$(wc -l < "$MLIR_TRANSFER")
if [ "$mlir_lines" -gt 1050 ]; then
    echo "❌ FAIL: MLIRTransfer.fs is $mlir_lines lines (expected ~965)"
    echo "   → Code may have been added that should be in witnesses"
    ((VIOLATIONS++))
else
    echo "✅ PASS: MLIRTransfer.fs size is acceptable ($mlir_lines lines)"
fi

# Check for central dispatch pattern
if grep -r "match.*with" "$MLIR_TRANSFER" | grep -q "witnessIntrinsic\|witnessOperation"; then
    echo "❌ FAIL: Central dispatch pattern found in MLIRTransfer.fs"
    echo "   → Witnesses should be called, not implemented inline"
    ((VIOLATIONS++))
else
    echo "✅ PASS: No central dispatch pattern found"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4: CallDispatch.fs Doesn't Exist (Should be .REMOVED)
# ═══════════════════════════════════════════════════════════════════════════

echo "━━━ CHECK 4: CallDispatch.fs Status ━━━"
echo ""

if [ -f "$WITNESS_DIR/CallDispatch.fs" ]; then
    call_dispatch_lines=$(wc -l < "$WITNESS_DIR/CallDispatch.fs")
    if [ "$call_dispatch_lines" -gt 300 ]; then
        echo "❌ FAIL: CallDispatch.fs exists and is $call_dispatch_lines lines"
        echo "   → Should be factored into focused witnesses"
        ((VIOLATIONS++))
    else
        echo "⚠️  WARN: CallDispatch.fs exists but is $call_dispatch_lines lines (acceptable)"
    fi
elif [ -f "$WITNESS_DIR/CallDispatch.fs.REMOVED" ]; then
    echo "✅ PASS: CallDispatch.fs is archived (.REMOVED)"
else
    echo "⚠️  WARN: CallDispatch.fs not found (may not be created yet)"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════════════════"
echo "Summary"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ "$VIOLATIONS" -eq 0 ]; then
    echo "✅ All architecture checks passed!"
    echo ""
    exit 0
else
    echo "❌ $VIOLATIONS violation(s) found"
    echo ""
    echo "See Serena memories:"
    echo "  - call_dispatch_central_dispatcher_failure"
    echo "  - mlir_transfer_read_only_audit"
    echo "  - call_dispatch_factoring_plan"
    echo ""
    exit 1
fi
