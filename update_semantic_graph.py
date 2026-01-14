
import os

file_path = '../fsnative/src/Compiler/Checking.Native/SemanticGraph.fs'

with open(file_path, 'r') as f:
    content = f.read()

# 1. Add LambdaBodyRegion to RegionKind
old_region_kind = """    /// Match case body region (match case index, 0-based)
    | MatchCaseRegion of index: int"""
new_region_kind = """    /// Match case body region (match case index, 0-based)
    | MatchCaseRegion of index: int
    /// Lambda body region (function body)
    | LambdaBodyRegion"""

if old_region_kind in content:
    content = content.replace(old_region_kind, new_region_kind)
else:
    print("Error: Could not find RegionKind definition to update")
    exit(1)

# 2. Update foldWithSCFRegions to handle SemanticKind.Lambda
old_match_case = """                        // Match: scrutinee is evaluated first, then each case body is a region
                        // NOTE: Pattern bindings are children of Match, processed as part of case body traversal
                        | SemanticKind.Match (scrutineeId, cases), Some hook ->
                            let parentId = node.Id
                            // Scrutinee - walk normally (value to match against)
                            let state = walk state scrutineeId
                            // Each case body is a separate region
                            cases
                            |> List.fold (fun (state, idx) case ->
                                let state = hook.BeforeRegion state parentId (MatchCaseRegion idx)
                                // Walk optional guard
                                let state = 
                                    match case.Guard with
                                    | Some guardId -> walk state guardId
                                    | None -> state
                                // Walk case body
                                let state = walk state case.Body
                                let state = hook.AfterRegion state parentId (MatchCaseRegion idx)
                                (state, idx + 1)
                            ) (state, 0)
                            |> fst"""

new_match_case = """                        // Match: scrutinee is evaluated first, then each case body is a region
                        // NOTE: Pattern bindings are children of Match, processed as part of case body traversal
                        | SemanticKind.Match (scrutineeId, cases), Some hook ->
                            let parentId = node.Id
                            // Scrutinee - walk normally (value to match against)
                            let state = walk state scrutineeId
                            // Each case body is a separate region
                            cases
                            |> List.fold (fun (state, idx) case ->
                                let state = hook.BeforeRegion state parentId (MatchCaseRegion idx)
                                // Walk optional guard
                                let state = 
                                    match case.Guard with
                                    | Some guardId -> walk state guardId
                                    | None -> state
                                // Walk case body
                                let state = walk state case.Body
                                let state = hook.AfterRegion state parentId (MatchCaseRegion idx)
                                (state, idx + 1)
                            ) (state, 0)
                            |> fst

                        // Lambda: body is a region
                        | SemanticKind.Lambda (_params, bodyId), Some hook ->
                            let parentId = node.Id
                            // Lambda body is a region
                            let state = hook.BeforeRegion state parentId LambdaBodyRegion
                            let state = walk state bodyId
                            let state = hook.AfterRegion state parentId LambdaBodyRegion
                            state"""

if old_match_case in content:
    content = content.replace(old_match_case, new_match_case)
else:
    print("Error: Could not find foldWithSCFRegions Match case to update")
    # Try to debug why it didn't match (indentation?)
    # For now, just exit
    exit(1)

with open(file_path, 'w') as f:
    f.write(content)

print("Successfully updated SemanticGraph.fs")
