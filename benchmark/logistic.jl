#!/usr/bin/env julia

using StochasticCubicDescent
import Distributions
using Base.Test

srand(42)


function gen_predictors(N, n)
    const auto_precision = -0.3
    const sd_range = 2.

    precision_mat = eye(n)
    for i in 2:(n-1)
        precision_mat[i-1, i] = precision_mat[i+1,i] = auto_precision
    end
    if n > 1
        precision_mat[2, 1] = precision_mat[n - 1, n] = auto_precision
    end
    cov_mat = precision_mat^-1
    cov_mat = (cov_mat + cov_mat') / 2
    mean_vec = [sqrt(cov_mat[j,j]) * ((j - (n/2)) * sd_range / (n / 2)) for j in 1:n]
    design_dist_unrounded = Distributions.MvNormal(mean_vec, cov_mat)

    design_unrounded = Distributions.rand(design_dist_unrounded, N)
    (sign(design_unrounded) + 1) / 2
end


function gen_responses(x, w_true)
    n, N = size(x)
    ret = Array(Float64, N)

    for i in 1:N
        c_i = 1 / (1  + exp(-x[:, i]' * w_true))
        ret[i] = rand() > c_i ? 0.0 : 1.0
    end

    ret
end


function get_logistic_objective(x, z, b, bH, reg)
    n, N = size(x)

    ξ!(w, ξ) = ξ[:] = rand(1:N, b)

    val(w, ξ) = begin
        y = 0.0
        for i in ξ
            c_i = 1 / (1  + exp(-x[:, i]' * w))
            y -= z[i] > 0.5 ? log(c_i) : log(1 - c_i)
        end

        # make these unbiased estimates of the full objective
        y /= length(ξ)

        # a per-example regularizer
        y + 0.5 * reg * (w' * w)
    end

    grad!(w, ξ, gr) = begin
        fill!(gr, 0.0)

        for i in ξ
            c_i = 1 / (1  + exp(-x[:, i]' * w))
            ci_offset = c_i - z[i]
            gr .+= ci_offset * x[:, i]
        end

        # make these unbiased estimates of the full objective
        gr ./= length(ξ)
        gr .+= reg * w
    end

    hv!(w, ξ, v, hv) = begin
        fill!(hv, 0.0)

        for i in ξ
            c_i = 1 / (1  + exp(-x[:, i]' * w))
            ci_ent = c_i * (1 - c_i)
            d_i = ci_ent * (x[:, i]' * v)
            hv .+= d_i * x[:, i]
        end

        hv ./= length(ξ)
        hv .+= reg * v
    end

    ObjectiveFunction(val, grad!, hv!, ξ!, 45.)
end


function run()
    const N = 7000
    const n = 40

    const reg = 1e2

    const b = 50
    const bH = 600

    const L = 1
    const M = 1

    x = gen_predictors(N, n)
    w_true = randn(n)
    z = gen_responses(x, w_true)
    f = get_logistic_objective(x, z, b, bH, reg)

    # verifying gradient
    grad1 = zeros(n)
    f1 = f.val(w_true, 44:55)
    f.grad!(w_true, 44:55, grad1)

    grad2 = zeros(n)
    f2 = f.val(w_true + 1e-5, 44:55)
    f.grad!(w_true + 1e-5, 44:55, grad2)

    finite_diff_deriv = (f2 - f1) / 1e-5
    @test sum(grad1) ≈ finite_diff_deriv atol=1e-1

    # verifying hessian
    finite_diff_hv = (grad2 - grad1) / 1e-5
    hv1 = zeros(n)
    f.hv!(w_true, 44:55, ones(n), hv1)
    @test hv1 ≈ finite_diff_hv atol=1e-1
end


run()
