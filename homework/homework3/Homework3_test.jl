using DMUStudent.HW3: HW3, DenseGridWorld, visualize_tree
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, initialstate, stateindex
using POMDPs
using D3Trees: inchrome, inbrowser
using StaticArrays: SA
using Statistics: mean, std
# using BenchmarkTools: @btime
using BenchmarkTools
using QuickPOMDPs: QuickPOMDP
using POMDPTools

function rollout(mdp, policy_function, s0, max_steps)
    r_total = 0.0                                   # Initialize û
    s = s0                                          # Initial state
    t = 0
    while !isterminal(mdp, s) && t < max_steps      # While not terminal and not max steps
        a = policy_function(mdp, s)                 # Sample action from policy
        s, r = @gen(:sp,:r)(mdp, s, a)              # Generate new state and reward
        r_total += discount(mdp)^t*r                # add discounted reward to total utility
        t += 1  
    end
    return r_total
end

function explore(A, s, N, Q, c)
    """
    # Arguments
    'A' : Action space of MDP (ie [:left, :right] or actions(mdp))
    's' : current state
    'N' : dictionary of state-action visit counts
    'Q' : dictionary of state-action value estimates
    'c' : exploration constant
    """
    Ns = sum([N[(s,a)] for a in A])
    if Ns == 0
        a_i =  argmax([Q[(s,a)] for a in A])
    else
        a_i = argmax([Q[(s,a)] + c*sqrt(log(Ns)/N[(s,a)]) for a in A])
    end
    return A[a_i]
end

function U_value_function(m,s)
    """
    U(s) = R(s) + γ*∑T(s'|s,a)U(s')
    """
    # R = reward_vectors(m)
    # R_expected = sum([R[a][stateindex(m, @gen(:sp,:r)(m, s, a)[1])] for a in actions(m)])
    # return R[:right][stateindex(m,s)] + R_expected        # All rewards are the same per action for this MDP, :right used for indexing
    T = [@gen(:sp,:r)(m, s, a)[1] for a in actions(m)]
    instR2 = @gen(:sp,:r)(m, s, :up)[2]
    R_exp2 = sum([@gen(:sp,:r)(m, sp, :up)[2] for sp in T])
    return instR2 + R_exp2
end

# function simulate!(π::MonteCarloTreeSearch, s, d=10)
function simulate!(mdp, s, d, c, N, Q, T)
    """
    # Arguments
    'mdp' : MDP model (ie DenseGridWorld)
    's'   : current state
    'd'   : depth of search
    'c'   : exploration constant
    'N'   : dictionary of state-action visit counts
    'Q'   : dictionary of state-action value estimates
    'T'   : dictionary of state-action-state transition counts
    """

    A = actions(mdp)                # get action space
    γ = discount(mdp)               # get discount factor

    if d ≤ 0
        return U_value_function(mdp, s)
    end

    # 2. Expansion
    if !haskey(N, (s, first(A)))    # initialize dictionary enties for state-action pairs of current state if not already present
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end

        # 3. Value Estimate
        return U_value_function(mdp, s)
    end

    # 1. Serach
    a = explore(A, s, N, Q, c)
    sp, r = @gen(:sp,:r)(mdp, s, a)   # TR(s,a)
    q = r + γ*simulate!(mdp, sp, d-1, c, N, Q, T)

    # 4. Back up
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    haskey(T, (s, a, sp)) ? (T[(s,a,sp)] += 1) : (T[(s,a,sp)] = 1)
    return q
end

function mcts(mdp, init_state, q, n, t, max_iters=7, depth=10)
    for _ in 1:max_iters
        simulate!(mdp, init_state, depth, 100, n, q, t)
    end
end

function select_action(m, s)

    start = time_ns()
    n = Dict{Tuple{statetype(m), actiontype(m)}, Int}()
    q = Dict{Tuple{statetype(m), actiontype(m)}, Float64}()
    t = Dict{Tuple{statetype(m), actiontype(m), statetype(m)}, Int}()

    for _ in 1:1000
    # while time_ns() < start + 40_000_000 # you can replace the above line with this if you want to limit this loop to run within 40ms
        simulate!(m, s, 10, 100, n, q, t) # replace this with mcts iterations to fill n and q
    end

    # select a good action based on q and/or n

    return actions(m)[argmax([q[(s,a)] for a in actions(m)])] # this dummy function returns a random action, but you should return your selected action
end

m = DenseGridWorld(seed=4)

@btime select_action(m, SA[35,35]) # you can use this to see how much time your function takes to run. A good time is 10-20ms.

# @benchmark select_action(m, SA[35,35]) 

# @profview select_action(m, SA[35,35])

@show results = [rollout(m, select_action, rand(initialstate(m)), 100) for _ in 1:100]
@show mean(results)
@show std(results) / sqrt(length(results))