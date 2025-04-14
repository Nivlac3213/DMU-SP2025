using CommonRLInterface
using Flux
using CommonRLInterface.Wrappers: QuickWrapper
using DMUStudent.HW5: HW5, mc


# Override to a discrete action space, and position and velocity observations rather than the matrix.
env = QuickWrapper(HW5.mc,
                   actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
                   observe=mc->observe(mc)[1:2]
                  )

# create your loss function for Q training here
function loss(Q, Q_target, s, a_ind, r, sp, done)
    γ = 0.99
    q_sa = Q(s)[a_ind]
    max_q_sp = maximum(Q_target(sp))
    target = r + (done ? 0.0 : γ * max_q_sp)
    return (target - q_sa)^2
end

function quick_evaluate(Q, env; n_episodes=500, max_steps_per_episode=500)
    total_reward = 0.0

    for _ in 1:n_episodes
        reset!(env)
        s = observe(env)
        episode_reward = 0.0

        for _ in 1:max_steps_per_episode
            a_ind = argmax(Q(s))
            a = actions(env)[a_ind]
            r = act!(env, a)
            s = observe(env)
            episode_reward += r

            if terminated(env)
                break
            end
        end

        total_reward += episode_reward
    end

    return total_reward / n_episodes
end

function epsilon_greedy(Q, s, ε, actions)
    if rand() < ε
        return rand(1:length(actions))  # random action index
    else
        return argmax(Q(s))  # index of best action
    end
end

function dqn(env)

    # Hyperparameters
    n_steps = 100_000
    γ = 0.99
    ε = 1           # starting epsilon
    ε_min = 0.05
    decay = 1e-5
    target_update = 1000
    eval_interval = 1000
    minibatch_size = 32

    # This network should work for the Q function - an input is a state; the output is a vector containing the Q-values for each action 
    Q = Chain(Dense(2, 128, relu),
              Dense(128, length(actions(env))))
    Q_target = deepcopy(Q)
    opt = Flux.setup(ADAM(0.0005), Q)

    buffer = []
    best_score = -Inf
    best_Q = deepcopy(Q)

    for step in 1:n_steps

        s = observe(env)
        a_ind = epsilon_greedy(Q, s, ε, actions(env))
        a = actions(env)[a_ind]
        r = act!(env, a)
        sp = observe(env)
        done = terminated(env)

        # if step % 100 == 0
        #     println("step $step | done = $done | r = $r")
        # end

        # Store experience
        push!(buffer, (s, a_ind, r, sp, done))
        # if length(buffer) > 10_000
        #     popfirst!(buffer)  # optional buffer size limit
        # end

        # Training step if buffer is warm
        if length(buffer) ≥ minibatch_size
            for data in rand(buffer, minibatch_size)
                loss_value, grads = Flux.withgradient(loss, Q, Q_target, data...)
                Flux.update!(opt, Q, grads[1])
            end
        end

        # Update target Q-network
        if step % target_update == 0
            Q_target = deepcopy(Q)
        end

        ε = max(ε_min, ε - decay)  # linear decay

        # Periodic evaluation
        if step % eval_interval == 0
            result = HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], n_episodes=100)
            score = result.score
            print(score)
            println("Step $step | ε = $(round(ε, digits=3)) | Score: $(round(score, digits=2))")

            if score > best_score
                best_score = score
                best_Q = deepcopy(Q)
                if score > 40
                    # Save JSON submission file
                    fname = string(step, "_score.json")
                    HW5.evaluate(s -> actions(env)[argmax(Q(s[1:2]))], "carson.anderson@colorado.edu"; fname=fname)
                    println("🔥 Saved submission at step $step with score $score")
                end
            end
        end

        if done
            reset!(env)
            s = observe(env)
        else
            s = sp
        end
    end

    return best_Q

end

Q = dqn(env)

HW5.evaluate(s->actions(env)[argmax(Q(s[1:2]))], n_episodes=1000) # you will need to remove the n_episodes=100 keyword argument and add your email as a positional argument to create a json file; evaluate needs to run 10_000 episodes to produce a json