"""
Quadratic objective
    x → 0.5 xᵀQx - bᵀx
"""
struct Quadratic{matT,vecT}
    Q::matT
    b::vecT
end

"Evaluate the quadratic objective."
(q::Quadratic)(x) = 0.5*dot(x,q.Q*x) - dot(q.b,x)

"Evaluate the quaratic gradient."
grad(q::Quadratic, x) = q.Q*x - q.b

struct LBFGSBQuad
    optimizer::L_BFGS_B
    bounds::Matrix{Float64}
    Q::Quadratic
    bl::Vector{Float64}
    bu::Vector{Float64}
    x0::Vector{Float64}
    m::Int64
    function LBFGSBQuad(Q::Quadratic, bl::Vector, bu::Vector; m::Integer=5)
        n = length(bl)
        bounds = Matrix{Float64}(undef, 3, n)
        optimizer = L_BFGS_B(n, m)
        bounds[1,:] = fill(2, 1, length(bl))
        bounds[2,:] = bl'
        bounds[3,:] = bu'
        x0 = (bl+bu)/2
        new(optimizer, bounds, Q, bl, bu, x0, m)
    end
end

""" Solve min_{l ≤ x ≤ u} 0.5x^TQx - b^Tx using LBFGSB."""
function boxquad(lss::LBFGSBQuad;
                 factr=1.0, pgtol=1e-6, iprint=-1,
                 maxfun=15000, maxiter=15000)
    Q, bounds, bl, bu, x0 = lss.Q, lss.bounds, lss.bl, lss.bu, lss.x0
    f, x = lss.optimizer(Q, (z,x) -> z.=grad(Q, x), x0, bounds,
                         m=lss.m, factr=factr, pgtol=pgtol, iprint=iprint,
                         maxfun=maxfun, maxiter=maxiter)
    return x
end

"""
    Solve the QP
    0.5*x'*Q*x + b'*x s.t. x_i = v_i for i in mask.
"""
function _solve_eq_qp(Q::Quadratic, mask, values)
    Q_new = _fix_values(Q, mask, values)
    x_free = Q_new.Q\Q_new.b
    x = deepcopy(values)
    ind_free = 1
    for i in eachindex(x)
        if !mask[i]
            x[i] = x_free[ind_free]
            ind_free+=1
        end
    end
    x
end


function _fix_values(Q_st::Quadratic, mask, values)
     n = length(Q_st.b)
     Q = Q_st.Q
     not_mask = (!).(mask)
     Q_new = Q[not_mask, not_mask]
     b_new = Q_st.b[not_mask]- Q[not_mask,:]*values #?????
     Quadratic(Q_new, b_new)
end

"""Attempts to solve min 0.5 x^TQx +b^Tx s.t. l ≤ x ≤ u by (repeatedly) guessing the active set."""
function active_set_min_bound_constrained_quadratic(Q_st :: Quadratic, l, u, max_iters = 5, tol=1E-10; mask_l = falses(length(l)), mask_u = falses(length(u)))
    Q,b = Q_st.Q, -Q_st.b
    n = length(b)
    eq_mask = falses(n)
    values = zeros(n)
    #if mask_l[i], x_i is constrained to be l_i
    #if mask_u[i], x_i is constrained to be u_i

    for iter in 1:max_iters
        # set up and solve equality constrained QP

        eq_mask .= false
        values .= 0.0
        for i in 1:n
            if mask_u[i]
                eq_mask[i] = true
                values[i] = u[i]
            elseif mask_l[i]
                eq_mask[i] = true
                values[i] = l[i]
            end
        end

        x = _solve_eq_qp(Q_st, eq_mask, values)

        #compute gradient
        g = grad(Q_st, x)
        #check kkt conditions
        if all(((x,l,u),) -> l-tol < x < u+tol, zip(x, l, u)) # primal feasibility
            ### compute dual variables by enforcing stationarity...
            # and check complementary slackness
            comp_slack = true
            for i in eachindex(x)
                if g[i] < -tol
                    if abs(x[i] - u[i]) > tol
                        comp_slack = false
                        mask_u[i] = false
                        mask_l[i] = false
                    end
                elseif g[i] > tol
                    if abs(x[i] - l[i]) > tol
                        comp_slack = false
                        mask_u[i] = false
                        mask_l[i] = false
                    end
                end
            end
            if comp_slack
                return (x, iter)
            end
        else # Not primal feasible, add constraints...
            for i in 1:n
                if x[i] < l[i] - tol
                    mask_l[i] = true
                    mask_u[i] = false
                end
                if x[i] > u[i] + tol
                    mask_u[i] = true
                    mask_l[i] = false
                end
            end
        end
    end
    return (zeros(n), 0)
end

function form_quadratic_from_least_squares(A,b)
    Quadratic(A'*A, A'*b)
end
