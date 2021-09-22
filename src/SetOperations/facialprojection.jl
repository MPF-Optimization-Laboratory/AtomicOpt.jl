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
function face_project(M::AbstractMatrix, F::AbstractFace, b::Vector)
    MF = M*F
    # println("finish create the linear map")
    # c = cg(MF'*MF, MF'*b)
    c = lsmr(MF, b)
    # println("finish solving the linear inverse problem")
    r = b - MF*c
    return c, r
end

