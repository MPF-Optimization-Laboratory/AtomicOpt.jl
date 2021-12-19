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
function face_project(M::AbstractLinearOp, F::AbstractFace, b::Vector)
    MF = M*F
    # println("finish create the linear map")
    # c = cg(MF'*MF, MF'*b)
    c = lsmr(MF, b)
    # println("finish solving the linear inverse problem")
    r = b - MF*c
    return c, r
end

function face_project_screening(Mop::MaskOP, F::NucBallFace, b::Vector{Float64})
    nnz, I, J, M = Mop.nnz, Mop.I, Mop.I, Mop.M
    U, V = F.U, F.V
    r = F.r
    W = Variable(r,r)

    UW = U*W
    s = 0.0
    for k = 1:nnz
        i, j = I[k], J[k]
        s += square(UW[i,:]*V[j,:] - b[k])
    end

    prob = minimize(nuclearnorm(W), s <= 1e-1*nnz)
    solve!(prob, SCS.Optimizer)

    c = vec(W.value)
    MF = Mop*F
    r = b - MF*c
    return c, r
end

