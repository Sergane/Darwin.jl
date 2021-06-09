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

##

using HDF5
using LinearAlgebra: diagm
using ProgressMeter
using SpecialFunctions: erfinv
using OffsetArrays

## Сетка:

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

include("configuration.jl")

## Частицы и их начальное распределение:

using Distributions

rand_uniform(L, N) = rand(Uniform(0,L), N)
rand_normal(σ, N) = rand(Normal(0,σ), N)

# function init_rand!(s, L, σˣ, σʸ, σᶻ)
# 	s.x .= rand_uniform(L, s.N)
# 	s.px .= rand_normal(σˣ, s.N)
# 	s.py .= rand_normal(σʸ, s.N)
# 	s.pz .= rand_normal(σᶻ, s.N)
# 	return
# end

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
	Vth = params.V_params .* √(1/set.m)
	set.px .= init_V(Vth[1], set.N) .* set.m
	set.py .= init_V(Vth[2], set.N) .* set.m
	set.pz .= init_V(Vth[3], set.N) .* set.m
	init_R = getfield(Main, Symbol("rand_", params.R_distribution))
	set.x .= init_R(L, set.N)

	set
end

##

function collect_sources!(φ, A, grid, species::Vector{<:ParticleSet})
	φ.ρ .= 0
	A.μ .= 0
	A.fy .= 0
	A.fz .= 0

	ρ = similar(φ.ρ)
	μ = similar(A.μ)
	fy = similar(A.fy)
	fz = similar(A.fz)
	for particle in species
		collect_sources!(ρ, μ, fy, fz, grid, particle)
		φ.ρ .+= ρ
		A.μ .+= μ
		A.fy .+= fy
		A.fz .+= fz
	end

	φ.ρ .-= mean_charge(species, grid)  # ионный фон

	boundary_condition!(φ.ρ)
	boundary_condition!(A.μ)
	boundary_condition!(A.fy)
	boundary_condition!(A.fz)
	# interpolation_bc!(A.ρ)
end

function collect_sources!(ρ, μ, fy, fz, grid, particle::ParticleSet)
	ρ .= 0
	μ .= 0
	fy .= 0
	fz .= 0
	@inbounds for k in 1:particle.N
		j, l = grid(particle.x[k])
		r = 1 - l
		ρ[j]    += l
		ρ[j+1]  += r
		fy[j]   += l*particle.py[k]
		fy[j+1] += r*particle.py[k]
		fz[j]   += l*particle.pz[k]
		fz[j+1] += r*particle.pz[k]
	end

	PPC = particle.PPC
	q = particle.q
	q_m = q/particle.m

	ρ .*= q/PPC
	μ .= ρ .* (q_m/PPC)
	fy .*= q_m/PPC
	fz .*= q_m/PPC
end

@inline function mean_charge(species, grid)
	Q_sum = sum(particles.q * particles.PPC for particles in species)
	PPC_sum = sum(particles.PPC for particles in species)
	Q_sum / PPC_sum
end

##

function scalar_potential!(φ, g)
	h² = g.h^2
	φ.x[1] = 0
	@simd for k in 1:g.N
		@inbounds φ.x[1] += k*φ.ρ[k]
	end
	φ.x[1] *= -h²/g.N
	φ.x[2] = -h²*φ.ρ[1] + 2φ.x[1]
	for k in 3:g.N
		@inbounds φ.x[k] = -h²*φ.ρ[k-1] + 2φ.x[k-1] - φ.x[k-2]
	end
	interpolation_bc!(φ.x)
end

function gradient!(to, f, g)
    for k in 1:g.N
        @inbounds to[k] = (f[k+1]-f[k-1]) / 2g.h
    end
    interpolation_bc!(to)
end

function vector_potential!(A, g)
	N = g.N
	h² = g.h^2
    M = diagm(0=>-h².*A.μ[1:N].-2, 1=>ones(N-1), -1=>ones(N-1))
	M[N,1] = M[1,N] = 1
	A.y[1:N] .= M \ (-h²*A.fy[1:N])
	A.z[1:N] .= M \ (-h²*A.fz[1:N])
	interpolation_bc!(A.y)
	interpolation_bc!(A.z)
end

function curl!(B, A, g)
    for k in g.in
        B.y[k] = -(A.z[k+1]-A.z[k-1]) / 2g.h
        B.z[k] =  (A.y[k+1]-A.y[k-1]) / 2g.h
    end
    interpolation_bc!(B.y)
    interpolation_bc!(B.z)
end

function field_energy(f, g)
    sum(f[g.in].^2)/(2*g.N)
end

function field_energy(v, i, g)
    sum(v[i,g.in].^2)/(2*g.N)
end

function kinetic_energy(s, A, g)
	Kx = 0.0
	Ky = 0.0
	Kz = 0.0
	for j in 1:s.N
		i, l = g(s.x[j])
		r = 1-l
		Kx +=  s.px[j]^2
		Ky += (s.py[j] - s.q*(l*A.y[i]+r*A.y[i+1]))^2
		Kz += (s.pz[j] - s.q*(l*A.z[i]+r*A.z[i+1]))^2
	end
	Kx/(2*s.m*s.N), Ky/(2*s.m*s.N), Kz/(2*s.m*s.N)
end

function current_densities!(Jx, Jy, Jz, A, e, g)
	Jx .= 0.0
	Jy .= 0.0
	Jz .= 0.0
	@inbounds for k in 1:e.N
		j, l = g(e.x[k])
		r = 1 - l
		Jx[j]   += l*e.px[k]
		Jx[j+1] += r*e.px[k]
		Jy[j] 	+= l*(e.py[k] - e.q*(l*A.y[j]+r*A.y[j+1]))
		Jy[j+1] += r*(e.py[k] - e.q*(l*A.y[j]+r*A.y[j+1]))
		Jz[j]   += l*(e.pz[k] - e.q*(l*A.z[j]+r*A.z[j+1]))
		Jz[j+1] += r*(e.pz[k] - e.q*(l*A.z[j]+r*A.z[j+1]))
	end
	Jx .*= e.q/(e.m*g.Npc)
	Jy .*= e.q/(e.m*g.Npc)
	Jz .*= e.q/(e.m*g.Npc)
	boundary_condition!(Jx)
	boundary_condition!(Jy)
	boundary_condition!(Jz)
end


@inline it(E, l, j) = (l*E[j]+(1-l)*E[j+1])
# @inline it(A, l, i, j) = (l*A[i,j]+(1-l)*A[i,j+1])


function leap_frog_halfstep!(s, dt, Ex, g)
	for k in eachindex(s.x)
		j, l = g(s.x[k])
		s.px[k] -= dt/2*s.q*(l*Ex[j]+(1-l)*Ex[j+1])
	end
end


function leap_frog!(s, dt, Ex, B, A, g)
	q = s.q
	m = s.m
	for k in eachindex(s.x)
		j, l = g(s.x[k])
		s.px[k] += dt*q*it(Ex,l,j) + dt*q/m*(
			 (s.py[k]-q*it(A.y,l,j))*it(B.z,l,j) -
			 (s.pz[k]-q*it(A.z,l,j))*it(B.y,l,j))
		s.x[k] += dt*s.px[k]/m
	end
	boundary_condition!(s, g)
end


function prestart!(species, model, time)
	g = model.g
	φ = model.φ
	A = model.A
	Ex= model.Ex
	B = model.B
	dt = step(time)
	@showprogress 1 "Prestart... " for t in time
		collect_sources!(φ, A, g, species)
		scalar_potential!(φ, g)
		gradient!(Ex, φ.x, g)
		Ex .*= -1
		vector_potential!(A, g)
		curl!(B, A, g)
		for s in species
			leap_frog!(s, dt, Ex, B, A, g)
		end
	end
end


struct KineticEnergies{T}
	sum::Vector{T}
	x::Vector{T}
	y::Vector{T}
	z::Vector{T}

	function KineticEnergies{T}(N::Int) where {T<:Real}
		arrays = [zeros(T,N) for i in 1:fieldcount(KineticEnergies)]
		new(arrays...)
	end
end

struct FieldEnergies{T}
	Ex::Vector{T}
	By::Vector{T}
	Bz::Vector{T}

	function FieldEnergies{T}(N::Int) where {T<:Real}
		arrays = [zeros(T,N) for i in 1:fieldcount(FieldEnergies)]
		new(arrays...)
	end
end

struct Energies{T}
	K::Vector{KineticEnergies{T}}
	fields::FieldEnergies{T}

	function Energies{T}(N::Int, species_num::Int) where {T<:Real}
		new([KineticEnergies{T}(N) for i in 1:species_num],
			FieldEnergies{T}(N))
	end
end

struct Fields{T}
	rho::Matrix{T}
	phi::Matrix{T}
	Ay::Matrix{T}  # векторный потенциал
	Az::Matrix{T}
	Ex::Matrix{T}
	By::Matrix{T}
	Bz::Matrix{T}

	function Fields{T}(M, N) where {T<:Real}
		arrays = [zeros(T,M,N) for i in 1:fieldcount(Fields)]
		new(arrays...)
	end
end

struct Densities{T}
	x::Matrix{T}
	y::Matrix{T}
	z::Matrix{T}

	function Densities{T}(M, N) where {T<:Real}
		arrays = [zeros(T,M,N) for i in 1:fieldcount(Densities)]
		new(arrays...)
	end
end

function write_SoA(dir, obj)
	h5open(dir, "w") do file
		for prop_name in propertynames(obj)
			write(file, string(prop_name), getfield(obj, prop_name))
		end
	end
end

function simulation!(species, model, time, dir)
	g = model.g
	φ = model.φ
	A = model.A
	Ex= model.Ex
	B = model.B
	Jx = similar(g.range)
	Jy = similar(Jx)
	Jz = similar(Jx)
	dt = step(time)
	field = Fields{Float64}(g.N, length(time))
	J = [Densities{Float64}(g.N, length(time)) for i in eachindex(species)]
	energy = Energies{Float64}(length(time), length(species))
	@showprogress 1 "Computing..." for t in eachindex(time)
		collect_sources!(φ, A, g, species)
		scalar_potential!(φ, g)
		gradient!(Ex, φ.x, g)
		Ex .*= -1
		vector_potential!(A, g)
		curl!(B, A, g)
		for (i,s) in enumerate(species)
			# current_densities!(J[i].x[:,t], J[i].y[:,t], J[i].z[:,t], A, s, g)
			leap_frog!(s, dt, Ex, B, A, g)
		end
		# сбор данных
		for (i,s) in enumerate(species)
			Kx, Ky, Kz = kinetic_energy(s, A, g)
			energy.K[i].sum[t] = Kx+Ky+Kz
			energy.K[i].x[t] = Kx
			energy.K[i].y[t] = Ky
			energy.K[i].z[t] = Kz
		end
		energy.fields.Ex[t] = field_energy(Ex, g)
		energy.fields.By[t] = field_energy(B.y, g)
		energy.fields.Bz[t] = field_energy(B.z, g)
		# fields_time[t] = t
		# fields_Jy[:,t] .= A.fy[g.in] .- A.μ[g.in].*A.y[g.in]
		# fields_Jz[:,t] .= A.fz[g.in] .- A.μ[g.in].*A.z[g.in]
		field.rho[:,t] .= φ.ρ[g.in]
		field.phi[:,t] .= φ.x[g.in]
		field.Ay[:,t] .= A.y[g.in]
		field.Az[:,t] .= A.z[g.in]
		field.Ex[:,t] .= Ex[g.in]
		field.By[:,t] .= B.y[g.in]
		field.Bz[:,t] .= B.z[g.in]
	end
	
	write_SoA(dir*"energies.h5", energy.fields)
	write_SoA(dir*"fields.h5", field)
	for i in eachindex(species)
		write_SoA(dir*"kinetic_energies_"*string(i)*".h5", energy.K[i])
		# write_SoA(dir*"J_"*string(i)*".h5", J[i])
	end
end

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

	g = Grid(params.L, params.cells_num)
	φ = ScalarPotential(g)
	A = VectorPotential(g)
	Ex = similar(g.range)
	B = MagneticField(g)
	NumericalModel{T}(species, g, φ, A, Ex, B)
end
NumericalModel(params) = NumericalModel{Float64}(params)

let
	config = Configuration("$(@__DIR__)/config.toml")
	println(config)
	params = NumericalParameters(config)
	model = NumericalModel(params)
	
	dir = isempty(ARGS) ? "test/" : ARGS[1]*"/"
	mkpath(dir)
	println("writing to ", abspath(dir))

	# TODO:
	# разбить модель на независимые абстракции
	# вывести здесь параметры модели 

	#step(time)*√(uˣ^2+uʸ^2+uᶻ^2) ≤ L/2Nc

	for set in model.species
		write_SoA(dir * "init_" * set.name * ".h5", set)
	end
	
	collect_sources!(model.φ, model.A, model.g, model.species)
	scalar_potential!(model.φ, model.g)
	gradient!(model.Ex, model.φ.x, model.g)
	model.Ex .*= -1
	vector_potential!(model.A, model.g)
	model.B.y .= 0
	model.B.z .= 0

	for s in model.species
		leap_frog_halfstep!(s, step(params.time), model.Ex, model.g)
	end

	# u = min(uˣ,uʸ,uᶻ)
	# for s in species
	# 	s.px .*= u/uˣ
	# 	s.py .*= u/uʸ
	# 	s.pz .*= u/uᶻ
	# end
	prestart!(model.species, model, params.time[1:params.prestart_steps])
	# for s in species
	# 	s.px .*= uˣ/u
	# 	s.py .*= uʸ/u
	# 	s.pz .*= uᶻ/u
	# end
	simulation!(model.species, model, params.time, dir)
end
