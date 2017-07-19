#!/usr/bin/env julia

using StochasticCubicDescent
using Base.Test


@testset "gradient descent on a cubic problem" begin
    # define a test function
    H0 = [1. 2.; 3. -4.]
    A = H0' * H0 - 10 * eye(2)  # the eigenvalues of A are -6.18 and 16.18
    b = [-1.6, -2.8]
    ρ = 3.3
    val(x) = 0.5x' * A * x + b' * x + (ρ / 3) * norm(x)^3
    grad!(x, grad) = grad[:] = A * x + b + ρ * norm(x) * x
    cubic_objective = ObjectiveFunction(val, grad!)

    # verify that a gradient is correct
    x = [5.0, 7.0]
    tmp = zeros(2)
    finite_diff_grad_sum = (val(x + 1e-4) - val(x)) / 1e-4
    analytic_grad_sum = sum(grad!(x, tmp))
    @test finite_diff_grad_sum ≈ analytic_grad_sum atol=1e-1

    # Below the step size and initial iterate are set according to
    #     "Gradient Descent Efficiently Finds the Cubic-Rgularized Non-Convex
    #      Newton Step" by Yair Carmon and John Duchi. 2017.
    β = norm(A)
    R = β / (2ρ) + sqrt((β / 2ρ)^2 + norm(b) / ρ)
    η = (8(β + ρ * R))^(-1)

    Rc_lower_bound = -β / ρ + R
    x0 = -Rc_lower_bound / norm(b) * b

    solver = GradientDescent(η, 1000, true, GDState[])
    state = GDState(x0)
    optimize!(solver, state, cubic_objective)

    @test norm(state.grad) ≈ 0.0 atol=1e-8

    # should hold by Lemma 2.2
    iterate_norms = [norm(st.x) for st in solver.history]
    iterate_increases = iterate_norms[2:end] - iterate_norms[1:end-1]
    @test all(iterate_increases .> 0)
end


@testset "Stochastic Cubic Descent" begin
    val(x, ξ) = 0.1x[1]^2 + exp(x[1] + 1) + x[1] * ξ[1]
    grad!(x, ξ, grad) = begin
        grad[1] = 0.2x[1] + exp(x[1] + 1) + ξ[1]
    end
    hv!(x, ξ, v, hv) = begin
        @assert x[1] < 5
        hv[1] = (0.2 + exp(x[1] + 1)) * v[1]
    end
    # very low noise
    ξ!(x, ξ) = ξ[:] = (rand(1) - 0.5) * 1e-4
    # exp(5) isn't a global Lipschitz constant, but it should hold
    # at every point that the algorithm visits
    f = ObjectiveFunction(val, grad!, hv!, ξ!, exp(5))

    solver = SCD(100)
    state = SCDState([-3.])

    optimize!(solver, state, f)

    @test norm(state.grad) ≈ 0.0 atol=1e-2
end
