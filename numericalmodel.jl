using OffsetArrays

struct Grid{T} <: AbstractVector{T}
	range::OffsetVector{T,<:AbstractRange{T}}
	in::Base.OneTo{Int64}
	h::T
	N::Int
	L::T
	function Grid{T}(L, N) where T
		h = L / N
		grid = range(-h/2, L+h/2, length=N+2)
      	new{T}(OffsetVector(grid, 0:N+1), Base.OneTo(N), step(grid), N, L)
  	end
end
Grid(L, N) = Grid{Float64}(L, N)
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

## Граничные условия

# @inline 
function interpolation_bc!(field)
    field[begin] = field[end-1]
    field[end] = field[begin+1]
    nothing
end

# @inline 
function interpolation_bc!(field, i)
    field[i,begin] = field[i,end-1]
	field[i,end] = field[i,begin+1]
    nothing
end

function boundary_condition!(s, g)
    for j in 1:s.N
		s.x[j] < 0   && (s.x[j] += g.L)
		s.x[j] > g.L && (s.x[j] -= g.L)
		s.x[j] < 0   && (s.x[j] = mod(s.x[j], g.L))
		s.x[j] > g.L && (s.x[j] = mod(s.x[j], g.L))
    end
end

function boundary_condition!(field)
    field[begin+1] += field[end]
    field[end-1]   += field[begin]
	nothing
end

## Частицы и их начальное распределение:

using Distributions

rand_uniform(L, N, _) = rand(Uniform(0,L), N)
rand_normal(σ, N, _) = rand(Normal(0,σ), N)

function hammersley(N, p)
	# p - простое число
	if p == 1
		return [1/2 : N-1/2 ...]/N
	end
	seq = zeros(N)
	for k = 1:N
		k!, p! = k, p
		while k! > 0
			a = k! % p
			seq[k] += a / p!
			k! = k! ÷ p
			p! *= p
		end
	end
	return seq
end
rand_uniform_quiet(L, N, p) = hammersley(N, p) * L

using SpecialFunctions: erfinv
function rand_normal_quiet(σ, N, p)
	σ * √2 * erfinv.(2 * hammersley(N, p) .- 1)
end

struct ParticleSet{T}
	x::Vector{T}
	px::Vector{T}
	py::Vector{T}
	pz::Vector{T}
	q::Float64
	m::Float64
	PPC::Int  # Particles Per Cell
	N::Int
	name::String

	function ParticleSet{T}(q, m, PPC, cells_num, name) where {T<:Real}
		N = PPC * cells_num
		new(zeros(T,N),
			zeros(T,N),
			zeros(T,N),
			zeros(T,N),
			q, m, PPC, N, name)
	end
end

function ParticleSet{T}(params::ParticleParameters, cells_num, L) where {T<:Real}
	set = ParticleSet{T}(params.charge, params.mass, params.ppc, cells_num, params.name)

	init_V = getfield(Main, Symbol("rand_", params.V_distribution))
	Vth = params.V_params .* √(1/set.m)  # V_params - "электронные" тепловые скорости
	set.px .= init_V(Vth[1], set.N, 3) .* set.m
	set.py .= init_V(Vth[2], set.N, 5) .* set.m
	set.pz .= init_V(Vth[3], set.N, 7) .* set.m
	init_R = getfield(Main, Symbol("rand_", params.R_distribution))
	set.x .= init_R(L, set.N, 2)

	set
end

##

# скалярный потенциал и сеточные величины, необходимые для его расчета
struct ScalarPotential{T}
	x::OffsetVector{T,Vector{T}}
	ρ::OffsetVector{T,Vector{T}}
end
function ScalarPotential{T}(N::Int) where T
	arrays = [OffsetVector(zeros(N+2), 0:N+1) for i in 1:fieldcount(ScalarPotential)]
	ScalarPotential{T}(arrays...)
end
ScalarPotential(N::Int) = ScalarPotential{Float64}(N)
function ScalarPotential{T}(g::Grid) where T
	arrays = [similar(g.range) for i in 1:fieldcount(ScalarPotential)]
	ScalarPotential{T}(arrays...)
end
ScalarPotential(g::Grid) = ScalarPotential{Float64}(g)

# векторный потенциал и сеточные величины, необходимые для его расчета
struct VectorPotential{T}
	y::OffsetVector{T,Vector{T}}
	z::OffsetVector{T,Vector{T}}
	μ::OffsetVector{T,Vector{T}}
	fy::OffsetVector{T,Vector{T}}
	fz::OffsetVector{T,Vector{T}}
end
function VectorPotential{T}(N::Int) where T
	arrays = [OffsetVector(zeros(N+2), 0:N+1) for i in 1:fieldcount(VectorPotential)]
	VectorPotential{T}(arrays...)
end
VectorPotential(N::Int) = VectorPotential{Float64}(N)
function VectorPotential{T}(g::Grid) where T
	arrays = [similar(g.range) for i in 1:fieldcount(VectorPotential)]
	VectorPotential{T}(arrays...)
end
VectorPotential(g::Grid) = VectorPotential{Float64}(g)

function Base.similar(A::VectorPotential{T}) where T
	arrays = [similar(A.μ) for i in 1:fieldcount(VectorPotential)]
	VectorPotential{T}(arrays...)
end

struct MagneticField{T}
	y::OffsetVector{T,Vector{T}}
	z::OffsetVector{T,Vector{T}}
end
function MagneticField{T}(N::Int) where T
	arrays = [OffsetVector(zeros(N+2), 0:N+1) for i in 1:fieldcount(MagneticField)]
	MagneticField{T}(arrays...)
end
MagneticField(N::Int) = MagneticField{Float64}(N)
function MagneticField{T}(g::Grid) where T
	arrays = [similar(g.range) for i in 1:fieldcount(MagneticField)]
	MagneticField{T}(arrays...)
end
MagneticField(g::Grid) = MagneticField{Float64}(g)

# минимальная модель, необходимая для вычисления одной итерации
struct NumericalModel{T}
	species::Vector{ParticleSet{T}}
	time::AbstractRange
	g::Grid{T}
	φ::ScalarPotential{T}
	A::VectorPotential{T}
	Ex::OffsetVector{T,Vector{T}}
	B::MagneticField{T}
end

function NumericalModel{T}(params::NumericalParameters) where T
	species = ParticleSet{T}[]
	for particle in params.species
		set = ParticleSet{Float64}(particle, params.cells_num, params.L)
		push!(species, set)
	end

	time = 0 : params.dt : params.T

	g = Grid(params.L, params.cells_num)
	φ = ScalarPotential(g)
	A = VectorPotential(g)
	Ex = similar(g.range)
	B = MagneticField(g)
	NumericalModel{T}(species, time, g, φ, A, Ex, B)
end
NumericalModel(params) = NumericalModel{Float64}(params)
