"""
Check that all elements of a vector have the same length.
Return the common length or throw a DimensionMismatch error.
"""
function checklength(atoms::Tuple{Vararg{AbstractAtom}}, msg)
    n = length(atoms[1])
    if !all(a->length(a)==n, atoms)
        throw(DimensionMismatch(msg))
    end
    return n
end

function checklength(sets::Tuple{Vararg{AbstractAtomicSet}}, msg)
    n = length(sets[1])
    if !all(A->length(A)==n, sets)
        throw(DimensionMismatch(msg))
    end
    return n
end

function checklength(faces::Tuple{Vararg{AbstractFace}}, msg)
    n = length(faces[1])
    if !all(F->length(F)==n, faces)
        throw(DimensionMismatch(msg))
    end
    return n
end
