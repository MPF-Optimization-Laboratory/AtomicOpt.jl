"""
Check that all elements of a vector have the same length.
Return the common length or throw a DimensionMismatch error.
"""
function checklength(list, msg)
    n = length(list[1])
    if !all(a->length(a)==n, list)
        throw(DimensionMismatch(msg))
    end
    return n
end
