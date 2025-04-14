
# ╔═╡ 1256a862-f61d-11ef-05e2-191ed73722f1
using PlutoUI, Plots, Flux

# ╔═╡ 82f8b78d-238d-4656-949f-eaf5b610fba4
function get_params()
	return PlutoUI.combine() do Child
		pairs = (
			("learning_rate", 10.0 .^ (-5:0), 1e-3),
			("n_epochs", 2_000:2_000:10_000, 10_000),
			("minibatch_size", [1, 10, 100], 1)
		)
		
		inputs = [
			md""" $(name): $(Child(name, Slider(vals; default)))"""
			for (name, vals, default) in pairs
		]
	
		md"""
		### Parameters
		$(inputs)
		"""
	end
end

# ╔═╡ 0a8b7c3b-7419-45ab-842f-4bee623f9aba
loss(model, x, y) = sum((model(x) - y) .^ 2) / length(y)

# ╔═╡ 116e7b6e-1af7-4a7b-b5ce-eaa608c7f02d
function train(x_data, y_data; 
	learning_rate=1e-3, n_epochs=1_000, save_every=50, minibatch_size=1
	)
	
	model = Chain(
		Dense(1=>50, tanh), 
		Dense(50=>50, tanh), 
		Dense(50=>1)
	)
	opt_state = Flux.setup(Adam(learning_rate), model)

	losses = Float32[]
	models = [deepcopy(model)]

	n_minibatches = length(y_data) ÷ minibatch_size
	
	for epoch in 1:n_epochs
		batch_loss = zero(Float32)

		for i in 1:n_minibatches
			idxs = (1:minibatch_size) .+ minibatch_size * (i - 1)
			x_minibatch = x_data[:, idxs]
			y_minibatch = y_data[:, idxs]

			function minibatch_objective(model)
				return loss(model, x_minibatch, y_minibatch)
			end
			
			minibatch_loss, grads = Flux.withgradient(minibatch_objective, model)
			
			Flux.update!(opt_state, model, grads[1])
			
			batch_loss += minibatch_loss / n_minibatches
		end
		
		push!(losses, batch_loss)

		if epoch % save_every == 0
			push!(models, deepcopy(model))
		end
	end

	return models, losses
end

# ╔═╡ 35bb93e3-2a76-4836-881a-d55f541c6954
begin
	x_true = range(0, 1, 500)
	y_true = sin.(4 * pi * x_true)
	p1 = plot(x_true, y_true; label="sin(4πx)", xlabel="x", ylabel="y")

	# Data generation
	# Each data entry should be its own column
	# Using Float32 is important as math will be faster compared to Float64
	n = 100
	x_data = rand(Float32, 1, n)
	y_data = sin.(4 * (pi * x_data)) + 0.1f0 * randn(Float32, 1, n)
	scatter!(p1, x_data[1,:], y_data[1,:]; label="Training Data", )
end

# ╔═╡ 5739d09b-2b73-4af3-814c-00d98b5b9b35
@bind params get_params()

# ╔═╡ eacb92a7-fe92-4db5-84d8-93016f790b6c
@info params;

# ╔═╡ b8b3ef90-2135-4088-8c0b-d42d107e9885
models, losses = train(x_data, y_data; params...);

# ╔═╡ 159670ea-2481-437c-9302-4812a06d35ac
@bind i Slider(1:length(models); default=length(models))

# ╔═╡ 84933c61-54f4-43ca-bf89-a9129c6c23d8
begin 
	p2 = plot(losses; 
		label=false, xlabel="Epochs", ylabel="Loss", 
		yaxis=:log, ylims=(1e-3,1e0)
	)
	p3 = scatter(p1, x_data[1,:], models[i](x_data)[1,:]; label="NN approx. ($i)")
	plot(p2, p3; layout=(2,1))
end

