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

## Частицы и их начальное распределение:

function hammersley(p, N)
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

using Distributions

uniform_rand(L, N) = rand(Uniform(0,L), N)
normal_rand(σ, N) = rand(Normal(0,σ),N)
uniform_quiet_H(p, L, N) = hammersley(p,N) * L

function normal_quiet_H(p, σ, N)
	√2σ*erfinv.(hammersley(p, N)*2 .- 1)
end

function init_rand!(s, L, σˣ, σʸ, σᶻ)
	s.x .= uniform_rand(L, s.N)
	s.px .= normal_rand(σˣ, s.N)
	s.py .= normal_rand(σʸ, s.N)
	s.pz .= normal_rand(σᶻ, s.N)
	return
end

init_rand!(s, L, σˣ, σʸ, σᶻ, ::Any) = init_rand!(s, L, σˣ, σʸ, σᶻ)

function init_quiet_H!(s, L, σˣ, σʸ, σᶻ, (p₁,p₂,p₃,p₄))
	s.x .= uniform_quiet_H(p₁, L, s.N)
	s.px .= normal_quiet_H(p₂, σˣ, s.N)
	s.py .= normal_quiet_H(p₃, σʸ, s.N)
	s.pz .= normal_quiet_H(p₄, σᶻ, s.N)
	return
end

struct ParticleSet{T}
	x::Vector{T}
	px::Vector{T}
	py::Vector{T}
	pz::Vector{T}
	q::Float64
	m::Float64
	N::UInt

	function ParticleSet{T}(q, m, N) where {T<:Real}
		new(zeros(T,N),
			zeros(T,N),
			zeros(T,N),
			zeros(T,N),
			q, m, N)
	end
end

function collect_sources!(ρ, A, g, particle::ParticleSet)
	rho = similar(ρ);  rho .= 0
	mu = similar(ρ);   mu  .= 0
	fy = similar(ρ);   fy  .= 0
	fz = similar(ρ);   fz  .= 0
	@inbounds for k in 1:particle.N
		j, l = g(particle.x[k])
		r = 1 - l
		rho[j]   += l
		rho[j+1] += r
		fy[j]   += l*particle.py[k]
		fy[j+1] += r*particle.py[k]
		fz[j]   += l*particle.pz[k]
		fz[j+1] += r*particle.pz[k]
	end
	q = particle.q
	q_m = q/particle.m
	rho .*= q/g.Npc
	mu .= rho .* (q_m/g.Npc)
	fy .*= q_m/g.Npc
	fz .*= q_m/g.Npc
	boundary_condition!(rho)
	rho .-= q  # ионный фон
	boundary_condition!(mu)
	boundary_condition!(fy)
	boundary_condition!(fz)
	interpolation_bc!(rho)
	ρ .+= rho
	A.μ .+= mu
	A.fy .+= fy
	A.fz .+= fz
end

function init_sources!(ρ, A)
	ρ .= 0
	A.μ .= 0
	A.fy .= 0
	A.fz .= 0
	return
end

function collect_sources!(ρ, A, g, species::Vector{<:ParticleSet})
	init_sources!(ρ, A)
	for particle in species
		collect_sources!(ρ, A, g, particle)
	end
end

function scalar_potential!(ϕ, ρ, g)
	h² = g.h^2
	ϕ[1] = 0
	@simd for k in 1:g.N
		@inbounds ϕ[1] += k*ρ[k]
	end
	ϕ[1] *= -h²/g.N
	ϕ[2] = -h²*ρ[1] + 2ϕ[1]
	for k in 3:g.N
		@inbounds ϕ[k] = -h²*ρ[k-1] + 2ϕ[k-1] - ϕ[k-2]
	end
	interpolation_bc!(ϕ)
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


begin
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


function prestart!(species, model, time, (uˣ,uʸ,uᶻ))
	u = min(uˣ,uʸ,uᶻ)
	dt = step(time)
	for s in species
		s.px .*= u/uˣ
		s.py .*= u/uʸ
		s.pz .*= u/uᶻ
	end
	@showprogress 1 "Prestart... " for t in time
		collect_sources!(model.ρ, model.A, model.g, species)
		scalar_potential!(model.φ, model.ρ, model.g)
		gradient!(model.Ex, model.φ, model.g)
		model.Ex .*= -1
		vector_potential!(model.A, model.g)
		curl!(model.B, model.A, model.g)
		for s in species
			leap_frog!(s, dt, model.Ex, model.B, model.A, model.g)
		end
	end
	for s in species
		s.px .*= uˣ/u
		s.py .*= uʸ/u
		s.pz .*= uᶻ/u
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
	ρ = model.ρ
	φ = model.φ
	Ex= model.Ex
	B = model.B
	A = model.A
	Jx = similar(ρ)
	Jy = similar(ρ)
	Jz = similar(ρ)
	dt = step(time)
	field = Fields{Float64}(g.N, length(time))
	J = [Densities{Float64}(g.N, length(time)) for i in eachindex(species)]
	energy = Energies{Float64}(length(time), length(species))
	@showprogress 1 "Computing..." for t in eachindex(time)
		init_sources!(ρ, A)
		for s in species
			collect_sources!(ρ, A, g, s)
		end
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
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
		field.rho[:,t] .= ρ[g.in]
		field.phi[:,t] .= φ[g.in]
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


## Параметры:

struct Model
	time::AbstractRange
	L::Float64
	Nc::Int
	Npc::Int
	uˣ::Float64
	uʸ::Float64
	uᶻ::Float64
	ion_bg::Bool
	init_method::String
	prestart_steps::Int
end

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
	g::Grid{T}
	ρ::OffsetVector{T,Vector{T}}
	φ::OffsetVector{T,Vector{T}}
	Ex::OffsetVector{T,Vector{T}}
	B::MagneticField{T}
	A::VectorPotential{T}
end

function NumericalModel{T}(g::Grid) where T
	ρ = similar(g.range)
	φ = similar(ρ)
	Ex = similar(ρ)
	B = MagneticField(g)
	A = VectorPotential(g)
	
	NumericalModel{T}(g, ρ, φ, Ex, B, A)
end
NumericalModel(g) = NumericalModel{Float64}(g)

let
	model = Model(0:0.25:50,
		5.24,
		256,
		1000,
		0.0316,
		0.0316,
		0.1,
		false,
		"rand",
		50)
	g = Grid(model.L, model.Nc, model.Npc)
	println("A: $((model.uᶻ/model.uˣ)^2-1)")
	
	dir = isempty(ARGS) ? "test/" : ARGS[1]*"/"
	mkpath(dir)
	println("writing to ", abspath(dir))

	# TODO:
	# разбить модель на независимые абстракции
	# вывести здесь параметры модели 

	#step(time)*√(uˣ^2+uʸ^2+uᶻ^2) ≤ L/2Nc

	N = g.Npc*g.N  # количество модельных частиц

	e = ParticleSet{Float64}(-1, 1, N)
	eval(Symbol("init_"*model.init_method*'!'))(e, model.L, (model.uˣ,model.uʸ,model.uᶻ)./√2..., (2,3,7,5))
	e.px .*= e.m
	e.py .*= e.m
	e.pz .*= e.m

	write_SoA(dir*"init_electron.h5", e)

	i = nothing
if !model.ion_bg
	i = ParticleSet{Float64}(1, 1836, N)
	K = √(e.m / i.m)
	eval(Symbol("init_"*model.init_method*'!'))(i, model.L, (model.uˣ,model.uʸ,model.uᶻ).*(K/√2)..., (2,3,7,5))
	i.px .*= i.m
	i.py .*= i.m
	i.pz .*= i.m

	write_SoA(dir*"init_ion.h5", i)
end
	species = [e]
	model.ion_bg || push!(species, i)

	num_model = NumericalModel(g)
	
	collect_sources!(num_model.ρ, num_model.A, num_model.g, species)
	scalar_potential!(num_model.φ, num_model.ρ, num_model.g)
	gradient!(num_model.Ex, num_model.φ, num_model.g)
	num_model.Ex .*= -1
	vector_potential!(num_model.A, num_model.g)
	num_model.B.y .= 0
	num_model.B.z .= 0

	for s in species
		leap_frog_halfstep!(s, step(model.time), num_model.Ex, num_model.g)
	end
	prestart!(species, num_model, model.time[1:model.prestart_steps], (model.uˣ, model.uʸ, model.uᶻ))
	simulation!(species, num_model, model.time, dir)
end
