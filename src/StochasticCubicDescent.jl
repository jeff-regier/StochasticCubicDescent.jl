module StochasticCubicDescent

export GradientDescent, GDState, optimize!, ObjectiveFunction

abstract type Method end
abstract type State end


struct ObjectiveFunction
    f::Function
    grad!::Function
    hv!::Function
end

struct GDState <: State
    iter::Ref{Int}
    x::Vector
    grad::Vector

    GDState(x) = new(0, x, similar(x))
end

# Gradient Descent
struct GradientDescent <: Method
    step_size::AbstractFloat
    store_history::Bool
    history::Vector{GDState}
    max_iters::Int

    GradientDescent(step_size, max_iters) = new(step_size, true, GDState[], max_iters)
end

function optimize!(method::GradientDescent,
                   state::GDState,
                   objective::ObjectiveFunction)
    for iter in 1:method.max_iters
        objective.grad!(state.x, state.grad)
        method.store_history && push!(method.history, deepcopy(state))
        state.iter[] += 1
        state.x .-= method.step_size * state.grad
    end
end

end # module
