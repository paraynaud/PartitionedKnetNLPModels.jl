# This partitioned chain assume a last layer that is a separable layer. 
# That way we ensure that there exists C = model.layers[end].out.

struct PartChainPSLDP <: PartitionedChain
	layers
	PartChainPSLDP(layers...) = new(layers)
end
(c :: PartChainPSLDP)(x) = (for l in c.layers; x = l(x); end; x)
(c :: PartitionedKnetNLPModels.PartChainPSLDP)(x) = (for (i,l) in enumerate(c.layers); println(i); x = l(x); end; x)

# fonction partitionnée
(c :: PartChainPSLDP)(d :: Knet.Data) = PartPSLDP(c; data=d, average=true)
(c :: PartChainPSLDP)(data :: Tuple{T1,T2}) where {T1,T2} = _PartPSLDP(c; data=data, average=true)

function PartPSLDP(model; data, dims=1, average=true, o...)	
	cnt = 0
	C = model.layers[end].out
	tmp = map(i -> CuArray(zeros(Float32, C)), 1:C)	
	for (x,y) in data
		(scores_pairs_classes, L) = PartPSLDP(model(x; o...), y; dims=dims, average=false)
		tmp += scores_pairs_classes
		cnt += L
	end
	average ? tmp ./ cnt : (tmp, cnt)
end
		
function _PartPSLDP(model; data, dims=1, average=true, o...)	
	(x,y) = data
	(scores_pairs_classes, L) = PartPSLDP(model(x; o...), y; dims=dims, average=false)
	acc = sum(sum.(scores_pairs_classes))
	average ? scores_pairs_classes ./ L : (acc, L)
end
function PartPSLDP(scores,labels :: AbstractArray{<:Integer}; dims=1, average=true)
	indices = findindices(scores, labels, dims=dims)
	losses = exp.(scores .- reshape(scores[indices], 1, length(indices))) # diminue par les scores par celui que l'on cherche à obtenir
	# absence de garantie < 1
	C = size(scores,1)
	acc = sum(losses)
	classes = (x-> x%C != 0 ? x%C : C ).(indices)
	indices_scores_classes = map( i -> findall(indice -> indice==i, classes), 1:C ) # select all the indices of each expected classes
	scores_pairs_classes = map(i -> vec(sum(losses[:, indices_scores_classes[i]], dims=2)), 1:C) # sum the loss of each pair (x,y) having the same y
	average ? scores_pairs_classes ./ length(labels) : (scores_pairs_classes, length(labels))
end

build_listes_indices(chain :: PartChainPSLDP) = build_listes_indices(PS(chain))
function build_listes_indices(ps_scores :: Vector{Vector{Int}}) 
	C = length(ps_scores)
	table_indices = reshape(map(i -> Vector{Int}(undef,0), 1:C^2), C, C)
	for i in 1:C
		for j in 1:C
			table_indices[i,j] = unique!(vcat(ps_scores[i], ps_scores[j]))
		end
	end
	return table_indices
end 


function partitioned_gradient(chain :: PartChainPSLDP, data_xy, table_indices :: Matrix{Vector{Int}}; type=Float32, n::Int=length(vcat_arrays_vector(vars)))
	vars = Knet.params(chain)		
	C = size(table_indices)[1]
	vector_grad_elt = Vector{Elemental_elt_vec{type}}(undef,0)
	tmp = similar(vars)
	for i in 1:C
		for j in 1:C
			if i != j 
				L = Knet.@diff chain(data_xy)[i][j]
				for (index,wᵢ) in enumerate(vars)
					tmp[index] = Param(Knet.grad(L,wᵢ))				
				end
				vec_tmp = vcat_arrays_vector(tmp)
				indices = table_indices[i,j]
				grad_elt = Vector(vec_tmp[indices])
				nie = length(indices)
				eev = PartitionedStructures.Elemental_elt_vec(grad_elt, indices, nie)
				push!(vector_grad_elt, eev)
			end
		end
	end 
	epv_grad = PartitionedStructures.create_epv(vector_grad_elt; n=n)
	return epv_grad
end 

function partitioned_gradient!(chain :: PartChainPSLDP, data_xy, table_indices :: Matrix{Vector{Int}}, epv_grad :: Elemental_pv{T}) where T <: Number
	vars = Knet.params(chain)		
	C = size(table_indices)[1]
	tmp = similar(vars)
	count = 0 
	for i in 1:C
		for j in 1:C
			if i != j 
				count += 1
				L = Knet.@diff chain(data_xy)[i][j]
				for (index,wᵢ) in enumerate(vars)
					tmp[index] = Param(Knet.grad(L,wᵢ))				
				end
				vec_tmp = vcat_arrays_vector(tmp)
				indices = table_indices[i,j]
				grad_elt = Vector(vec_tmp[indices])
				PartitionedStructures.set_eev!(epv_grad, count, grad_elt)				
			end
		end
	end 
	epv_grad
end 
