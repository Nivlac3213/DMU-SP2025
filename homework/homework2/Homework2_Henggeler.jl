using DMUStudent.HW2
using POMDPs: states, actions
using POMDPTools: ordered_states

##############
# Instructions
##############
#=

This starter code is here to show examples of how to use the HW2 code that you
can copy and paste into your homework code if you wish. It is not meant to be a
fill-in-the blank skeleton code, so the structure of your final submission may
differ from this considerably.

=#

############
# Question 3
############

@show actions(grid_world) # prints the actions. In this case each action is a Symbol. Use ?Symbol to find out more.

T = transition_matrices(grid_world)
display(T) # this is a Dict that contains a transition matrix for each action

@show T[:left][1, 2] # the probability of transitioning between states with indices 1 and 2 when taking action :left

R = reward_vectors(grid_world)
display(R) # this is a Dict that contains a reward vector for each action

@show R[:right][1] # the reward for taking action :right in the state with index 1

function value_iteration(m)
    """
    Value iteration algorithm for a POMDP algorithm: 
    1. Initialize U, U' differently
    2. While ||U-U'|| > ϵ                               While U & U' are sufficiently different (Alternativly maximum iterations)
    3.      U <-- U'                                    U gets U'
    4.      U'<-- max (R(s,a) + γ*∑ₛ T(s'|s,a)U(s')) ∀s ∈ S
       end
    5. return U
    """
    
    #initialize variables
    A   = actions(m)        # list of actions in MDP:m
    γ   = discount(m)       # discount factor
    S   = states(m)         # list of states in MDP:m
    T   = transition_matrices(m)
    R   = reward_vectors(m)
    k_max = 100             # maximum improvement iterations
    ϵ   = 1e-6              # value function convergence criteria
    
    # 1. Initialize U and U' 
    V   = rand(length(states(m)))
    Vp  = rand(length(states(m)))

    # 2. While ||V-V'|| > espsilon, Alternativly maximum iterations
    while (maximum(abs, V-Vp) > ϵ)       # Comment this line and uncomment the next line to run for a fixed number of iterations
    # for k in 1:k_max
        
        # 3. U <-- U' : U gets U'
        copy!(V,Vp)
        
        # 4. U' <-- max(R(s,a) + γ*∑T(s'|s,a)U(s')) ∀ s ∈ S
        for s in ordered_states(m)          # Iterate over all states in value function
            s_i = stateindex(m,s)           # Get state index of current state
            a_vec = []                      # place holder for value function results per action
            
            for a in A                          # Iterate over all actions
                td  = transition(m, s, a)           # Get non-zero state transition probabilities for current state-action pair
                # expected_reward = sum( T[a][s_i,stateindex(m,sp)] * V[stateindex(m,sp)] for sp in support(td) ) 
                expected_reward = sum( pdf(td,sp) * V[stateindex(m,sp)] for sp in support(td) ) 
                vs = R[a][s_i] + γ * expected_reward # Compute expected reward over possible next states
                push!(a_vec, vs)
            end

            # take max action result
            Vp[s_i] = maximum(a_vec)
        end

    end

    # 5. Return U 
    return Vp
end

# You can use the following commented code to display the value. If you are in an environment with multimedia capability (e.g. Jupyter, Pluto, VSCode, Juno), you can display the environment with the following commented code. From the REPL, you can use the ElectronDisplay package.
# display(render(grid_world, color=V))

############
# Question 4
############

# You can create an mdp object representing the problem with the following:
m = UnresponsiveACASMDP(2)

# transition_matrices and reward_vectors work the same as for grid_world, however this problem is much larger, so you will have to exploit the structure of the problem. In particular, you may find the docstring of transition_matrices helpful:
display(@doc(transition_matrices))

V = value_iteration(m)

@show HW2.evaluate(V)

########
# Extras
########

# The comments below are not needed for the homework, but may be helpful for interpreting the problems or getting a high score on the leaderboard.

# Both UnresponsiveACASMDP and grid_world implement the POMDPs.jl interface. You can find complete documentation here: https://juliapomdp.github.io/POMDPs.jl/stable/api/#Model-Functions

# To convert from physical states to indices in the transition function, use the stateindex function
# IMPORTANT NOTE: YOU ONLY NEED TO USE STATE INDICES FOR THIS ASSIGNMENT, using the states may help you make faster specialized code for the ACAS problem, but it is not required
using POMDPs: states, stateindex

s = first(states(m))
@show si = stateindex(m, s)

# To convert from a state index to a physical state in the ACAS MDP, use convert_s:
using POMDPs: convert_s

@show s = convert_s(ACASState, si, m)

# To visualize a state in the ACAS MDP, use
render(m, (s=s,))

# function backup(𝒫::MDP, U, s)
# return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
# end

# function lookahead(𝒫::MDP, U, s, a)
#    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
#    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
#    end

# function lookahead(𝒫::MDP, U::Vector, s, a)
#    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
#    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
# end
    
# function backup(𝒫::MDP, U, s)
#     return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
# end
    