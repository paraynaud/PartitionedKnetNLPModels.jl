using KnetNLPModels

"""
    create_minibatch(X, Y, minibatch_size)

Create a minibatch of the data `X`, `Y` of size `minibatch_size`.
"""
KnetNLPModels.create_minibatch(x_data, y_data, minibatch_size) = minibatch(x_data, y_data, minibatch_size; xsize=(size(x_data, 1), size(x_data, 2), 1, :))

"""
    vector_params(chain :: C) where C <: Chain

		vector_params(nlp :: PartitionedKnetNLPModel)

Retrieves the variables within `chain` or `nlp.chain` as a vector. 
"""
KnetNLPModels.vector_params(chain :: C) where C <: Chain = Array(vcat_arrays_vector(params(chain)))
KnetNLPModels.vector_params(nlp :: PartitionedKnetNLPModel) = nlp.w

"""
    vcat_arrays_vector(arrays_vector)

Flatten a vector of arrays to a vector. It concatenate the variables produce by applying `Knet.cat1d` to each array.
"""
KnetNLPModels.vcat_arrays_vector(arrays_vector) = vcat(Knet.cat1d.(arrays_vector)...)


""" 
    reset_minibatch_train!(nlp :: PartitionedKnetNLPModel{T, S, C}) where {T, S, C}

Take a new training minibatch for the `PartitionedKnetNLPModel`. Usually use before a new evaluation.
"""
KnetNLPModels.reset_minibatch_train!(nlp :: PartitionedKnetNLPModel{T, S, C}) where {T, S, C} = nlp.current_minibatch_training = rand(nlp.minibatch_train)

"""
    reset_minibatch_test!(nlp :: PartitionedKnetNLPModel{T, S, C}) where {T, S, C}

Take a new testing minibatch for the `PartitionedKnetNLPModel`. Usually use before a new accuracy test.
"""
KnetNLPModels.reset_minibatch_test!(nlp :: PartitionedKnetNLPModel{T, S, C}) where {T, S, C} = nlp.current_minibatch_testing = rand(nlp.minibatch_test)

""" 
    accuracy(nlp :: PartitionedKnetNLPModel{T, S, C}) where {T, S, C}

Computes the accuracy of the network `nlp.chain` given the data in `nlp.minibatch_test`.
The accuracy is based from the whole testing dataset.
"""
KnetNLPModels.accuracy(nlp :: PartitionedKnetNLPModel{T, S, C}) where {T, S, C} = Knet.accuracy(nlp.chain; data=nlp.minibatch_test)

"""
		build_layer_from_vec(v :: Vector{T}, var_layers :: CuArray{T, N, CUDA.Mem.DeviceBuffer} where N, index :: Int) where {T <: Number}

Inverse of the function `Knet.cat1d`, it builds a CuArray similar to `var_layers` from `vec`.
The return values are those of the vector in the range of `index+1:index+consumed_index`.
This method is not optimised, it consumes memory.
"""
function build_layer_from_vec(v :: Vector{T}, var_layers :: CuArray{T, N, CUDA.Mem.DeviceBuffer} where N, index :: Int) where {T <: Number}
  dims = ndims(var_layers)
  size_var_layers = size(var_layers)
  tmp_array = Array{T, dims}(undef, size_var_layers)	
  cuArray = CuArray(tmp_array)
  product_dims = build_layer_from_vec!(cuArray, v, index)
  return (cuArray, product_dims)
end

"""
		build_layer_from_vec!(cuArray :: CuArray{T, N, CUDA.Mem.DeviceBuffer} where N, v :: Vector{T}, index :: Int) where {T <: Number}

Inverse of the function `Knet.cat1d`, it sets `cuArray` to the values of `vec` in the range index+1:index+consumed_index.
"""
function build_layer_from_vec!(cuArray :: CuArray{T, N, CUDA.Mem.DeviceBuffer} where N, v :: Vector{T}, index :: Int) where {T <: Number}
	sizecuArray = reduce(*, size(cuArray))
	copyto!(cuArray, v[index+1:index+sizecuArray])
  return sizecuArray
end 

"""
		build_nested_array_from_vec(chain_ANN :: C, v :: Vector{T}) where {C <: Chain, T <: Number}

		build_nested_array_from_vec(model :: PartitionedKnetNLPModel{T, S, C}, v :: Vector{T}) where {T, S, C}
    
Build a vector of KnetArrays from `v` similar to `Knet.params(model.chain)` or 'Knet.params(chain_ANN)'.
It calls iteratively `build_layer_from_vec` to build each intermediary `CuArrays`.
This method is not optimised, it consumes memory.
"""
KnetNLPModels.build_nested_array_from_vec(model :: PartitionedKnetNLPModel{T, S, C}, v :: Vector{T}) where {T, S, C} = build_nested_array_from_vec(model.chain, v)
function KnetNLPModels.build_nested_array_from_vec(chain_ANN :: C, v :: Vector{T}) where {C <: Chain, T <: Number}
  param_chain = params(chain_ANN) # :: Param
  size_param = mapreduce((var_layer -> reduce(*, size(var_layer))), +, param_chain)
  size_param == length(v) || error("Dimension of Vector v mismatch, function rebuild_nested_array $(size_param) != $(length(v))")

  param_value = (x -> x.value).(param_chain) # :: Vector{KnetArrays}
  vec_CuArray = build_nested_array_from_vec(param_value, v) 
  return vec_CuArray
end

function KnetNLPModels.build_nested_array_from_vec(nested_array :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}, v :: Vector{T}) where {T <: Number}  
	vec_CuArray = map(i-> similar(nested_array[i]), 1:length(nested_array))
  build_nested_array_from_vec!(vec_CuArray, v)
  return vec_CuArray
end

"""
		build_nested_array_from_vec!(vec_CuArray :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}, v :: Vector{T}) where {T <: Number}

		build_nested_array_from_vec!(model :: PartitionedKnetNLPModel{T, S, C}, new_w :: Vector) where {T, S, C}
    
Build a vector of `CuArrays` from `v` similar to `Knet.params(model.chain)` or `Knet.params(chain_ANN)`.
It calls iteratively `build_layer_from_vec` to build each intermediary `CuArray`.
This method is not optimised, it consumes memory.
"""
KnetNLPModels.build_nested_array_from_vec!(model :: PartitionedKnetNLPModel{T, S, C}, new_w :: Vector) where {T, S, C} = build_nested_array_from_vec!(model.nested_cuArray, new_w)
function KnetNLPModels.build_nested_array_from_vec!(vec_CuArray :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}, v :: Vector{T}) where {T <: Number}
  index = 0
  for variable_layer in vec_CuArray
    consumed_indices = build_layer_from_vec!(variable_layer, v, index)
    index += consumed_indices
  end	
end

"""
		set_vars!(model :: PartitionedKnetNLPModel{T, S, C}, new_w :: Vector) where {T, S, C}

    set_vars!(chain_ANN :: C, nested_w :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}) where {C <: Chain, T <: Number}

Set the variables of `model` or `chain` to new_w.
Build a vector of `CuArrays` from `v` similar to `Knet.params(model.chain)`.
Then it sets these variables to the nested array.
"""
KnetNLPModels.set_vars!(vars :: Vector{Param}, new_w :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}) where {T<: Number} = map(i -> vars[i].value .= new_w[i], 1:length(vars))
KnetNLPModels.set_vars!(chain_ANN :: C, nested_w :: Vector{CuArray{T, N, CUDA.Mem.DeviceBuffer} where N}) where {C <: Chain, T <: Number} = set_vars!(params(chain_ANN), nested_w) 
function KnetNLPModels.set_vars!(model :: PartitionedKnetNLPModel{T, S, C}, new_w :: Vector) where {T, S, C}
	build_nested_array_from_vec!(model, new_w)	
  set_vars!(model.chain, model.nested_cuArray)
  model.w .= new_w
end