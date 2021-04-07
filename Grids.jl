module Grids

export Grid

using OffsetArrays

struct Grid{T} <: AbstractVector{T}
	range::OffsetVector{T,<:AbstractRange{T}}
	in::Base.OneTo{Int64}
	h::T
	N::Int
	Npc::Int
	L::T
	function Grid{T}(L, Nc, Npc) where T
		h = L / Nc
		grid = range(-h/2, L+h/2, length=Nc+2)
      	new{T}(OffsetVector(grid, 0:Nc+1), Base.OneTo(Nc), step(grid), Nc, Npc, L)
  	end
end
Grid(L, Nc, Npc) = Grid{Float64}(L, Nc, Npc)
Base.size(g::Grid) = size(g.range)
Base.axes(g::Grid) = axes(g.range)
Base.IndexStyle(::Type{<:Grid}) = IndexLinear()
function Base.getindex(g::Grid{T}, i::Int)::T where {T}
    getindex(g.range, i)
end
function Base.step(g::Grid{T})::T where {T}
    step(g.range)
end

@inline function (G::Grid)(r)
    i = unsafe_trunc(Int,r/G.h+0.5)
    (i, (G[i+1] - r)/G.h)
end

end # module
