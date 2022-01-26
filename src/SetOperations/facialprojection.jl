########################################################################
# Facial projection
########################################################################

"""
    face_project(M, F::AbstractFace, b::Vector)

Obtain the projection of the vector `b` onto the image of the
face `F` under the linear map `M`, i.e., ``MF``. In particular,
`c` solves the least-squares problem

    minimize_c  ½‖(MF)c-b‖²  subj to  Fc ∈ cone(F)

Each face `F` is aware of the required constraint of `c`.
"""
function face_project!(M::AbstractLinearOp, F::AbstractFace, b::Vector{Float64}, c::Vector{Float64}, r::Vector{Float64})
    # create linear map
    MF = M*F
    # solve the projection by lsmr
    lsmr!(c, MF, b)
    # compute residual
    r .= b - MF*c
    return nothing
end


