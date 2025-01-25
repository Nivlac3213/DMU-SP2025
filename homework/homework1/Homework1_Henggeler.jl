import DMUStudent.HW1
using DMUStudent.HW1

#------------- 
# Problem 4
#-------------

function f(a, bs)

    # Multiply all the vectors in 'bs' by 'a' and store in temp matrix
    result_vectors = [a * b for b in bs]  # Multiply each vector in `bs` by `a`
    @show result_vectors

    # Compute the elementwise maximum across all resulting vectors
    max_vector = Vector{eltype(a)}();
    for i in 1:length(bs[1])                    # iterate i'th term of all vectors
        max_i = nothing;                        # Reset max_i
        for j in 1:length(result_vectors)       # iterate through all j vectors
            if isnothing(max_i)                 
                max_i = result_vectors[j][i]    # init max_i for first term
            elseif result_vectors[j][i] > max_i # check if j'th term is max of all j'th terms
                max_i = result_vectors[j][i]    # Store new i'th elem max
            end
        end
        append!(max_vector, oftype(a[1][1],max_i));    # Append max i'th element to max_vector
    end

    return max_vector
end

#define vector 'a'
a = [1.0 2.0; 3.0 4.0];         #  [2.0 0.0; 0.0 1.0];
@show a

#define vector of vectors 'bs'
bs = [[1.0, 3.0], [3.0, 4.0], [5.0, 5.0]];  # [[1.0, 2.0], [2.0, 1.0]]
@show bs

# Apply Function f()
@show f(a, bs);

# Create JSON file for submission
HW1.evaluate(f, "calvin.henggeler@colorado.edu")
