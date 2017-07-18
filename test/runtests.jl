#!/usr/bin/env julia

using StochasticCubicDescent
using Base.Test


@testset "gradient descent on a cubic problem" begin
    # define a test function
    H0 = [1. 2.; 3. -4.]
    A = H0' * H0 - 10 * eye(2)  # the eigenvalues of A are -6.18 and 16.18
    b = [-1.6, -2.8]
    rho = 3.3
    value(x) = 0.5x' * A * x + b' * x + (rho / 3) * norm(x)^3
    grad!(x, grad) = begin
        grad[:] = A * x + b + rho * norm(x) * x
    end
    hv!(x, v, hv) = begin
        hv[:] = A * v + rho * norm(x) * v + (rho / norm(x)) * (x' * v) * x
    end
    cubic_objective = ObjectiveFunction(value, grad!, hv!)

    # verify that a gradient is correct
    x = [5.0, 7.0]
    tmp = zeros(2)
    finite_diff_grad_sum = (value(x + 1e-4) - value(x)) / 1e-4
    analytic_grad_sum = sum(grad!(x, tmp))
    @test finite_diff_grad_sum ≈ analytic_grad_sum atol=1e-1

    # verify that a Hessian-vector product is correct
    finite_diff_hv_sum = (grad!(x + 1e-4, tmp) - grad!(x, tmp)) / 1e-4
    analytic_hv_sum = hv!(x, ones(2), tmp)
    @test finite_diff_hv_sum ≈ analytic_hv_sum atol=1e-1

    # Below the step size and initial iterate are set according to
    #     "Gradient Descent Efficiently Finds the Cubic-Rgularized Non-Convex
    #      Newton Step" by Yair Carmon and John Duchi. 2017.
    gamma = -eigvals(A)[1]
    beta = norm(A)
    R = beta / (2rho) + sqrt((beta / 2rho)^2 + norm(b) / rho)
    eta = (8(beta + rho * R))^(-1)

    Rc_lower_bound = -beta / rho + R
    x0 = -Rc_lower_bound / norm(b) * b

    method = GradientDescent(eta, 1000)
    state = GDState(x0)
    optimize!(method, state, cubic_objective)

    @test norm(state.grad) ≈ 0.0 atol=1e-8

    # should hold by Lemma 2.2
    iterate_norms = [norm(st.x) for st in method.history]
    iterate_increases = iterate_norms[2:end] - iterate_norms[1:end-1]
    @test all(iterate_increases .> 0)
end
