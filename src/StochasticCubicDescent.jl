module StochasticCubicDescent

export optimize!, ObjectiveFunction, GradientDescent, GDState, SCD, SCDState


abstract type Solver end
abstract type State end

struct ObjectiveFunction
    val::Function     # evaluates the objective function
    grad!::Function   # evaluates the gradient
    hv!::Function     # hessian-vector product (may be left undefined)
    ξ!::Function      # the noise (may be undefined for deterministic functions)
    ρ::AbstractFloat  # a Lipschitz constant for the Hessian
end

noop() = None
ObjectiveFunction(val, grad!) = ObjectiveFunction(val, grad!, noop, noop, -1.0)


# Gradient Descent (GD)

struct GDState <: State
    iter::Ref{Int}
    x::Vector
    grad::Vector

    GDState(x) = new(0, x, similar(x))
end

struct GradientDescent <: Solver
    η::AbstractFloat
    max_iters::Int
    store_history::Bool
    history::Vector{GDState}
end

function GradientDescent(η, max_iters)
    GradientDescent(η, max_iters, false, GDState[])
end

function optimize!(solver::GradientDescent,
                   state::GDState,
                   f::ObjectiveFunction)
    while state.iter[] < solver.max_iters
        f.grad!(state.x, state.grad)
        solver.store_history && push!(solver.history, deepcopy(state))
        state.iter[] += 1
        state.x .-= solver.η * state.grad
    end
end


# Stochastic Cubic Descent (SCD)

struct SCDState <: State
    iter::Int
    x::Vector
    grad::Vector
    hv::Vector
    ξ::Vector

    SCDState(x0) = new(0, x0, similar(x0), similar(x0))
end

struct SCD <: Solver
    max_iters::Int
    store_history::Bool
    history::Vector{SCDState}
end

function SCD(max_iters)
    SCD(max_iters, false, SCDState[])
end

function optimize!(solver::SCD, state::SCDState, f::ObjectiveFunction)
    β = f.ρ  # Here f.ρ is an upper bound on the β from Carmon-Duchi 2017
    R = β / (2f.ρ) + sqrt((β / 2f.ρ)^2 + norm(b) / f.ρ)
    η = (4(β + f.ρ * R))^(-1)
    gd_solver = GradientDescent(η, 1000)

    Rc_lower_bound = -β / f.ρ + R
    gd_x0 = -Rc_lower_bound / norm(b) * b
    gd_state = GDState(gd_x0)

    val(x) = begin
        f.hv!(state.x + x, x, state.hv; ξ=state.ξ)
        0.5(x' * state.hv) + x' * state.grad + (f.ρ / 3) * norm(x)^3
    end
    grad!(x, grad) = begin
        f.hv!(state.x + x, x, state.hv; ξ=state.ξ)
        grad[:] = state.hv + state.grad + f.ρ * norm(x) * x
    end
    gd_subproblem = ObjectiveFunction(val, grad!)

    while state.iter[] < solver.max_iters
        f.grad!(state.x, state.grad)  # stochastic
        f.ξ!(state.x, state.ξ)
        solver.store_history && push!(solver.history, deepcopy(state))
        state.iter[] += 1
        optimize!(gd_solver, gd_state, gd_subproblem)
        state.x .+= gd_state.x
    end
end


end # module
