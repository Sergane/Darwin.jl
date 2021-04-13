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
	s isa Nothing && return
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


function prestart!(e, i, ρ, φ, Ex, B, A, g, time, u, (uˣ,uʸ,uᶻ))
	dt = step(time)
	e.px .*= u/uˣ
	e.py .*= u/uʸ
	e.pz .*= u/uᶻ
	i.px .*= u/uˣ
	i.py .*= u/uʸ
	i.pz .*= u/uᶻ
	@showprogress 1 "Prestart... " for t in time
		collect_sources!(ρ, A, g, [e, i])
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
		Ex .*= -1
		vector_potential!(A, g)
		curl!(B, A, g)
		leap_frog!(e, dt, Ex, B, A, g)
		leap_frog!(i, dt, Ex, B, A, g)
	end
	e.px .*= uˣ/u
	e.py .*= uʸ/u
	e.pz .*= uᶻ/u
	i.px .*= uˣ/u
	i.py .*= uʸ/u
	i.pz .*= uᶻ/u
	return
end


function prestart!(e, ::Nothing, ρ, φ, Ex, B, A, g, time, u, (uˣ,uʸ,uᶻ))
	dt = step(time)
	e.px .*= u/uˣ
	e.py .*= u/uʸ
	e.pz .*= u/uᶻ
	@showprogress 1 "Prestart... " for t in time
		collect_sources!(ρ, A, g, [e])
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
		Ex .*= -1
		vector_potential!(A, g)
		curl!(B, A, g)
		leap_frog!(e, dt, Ex, B, A, g)
	end
	e.px .*= uˣ/u
	e.py .*= uʸ/u
	e.pz .*= uᶻ/u
	return
end

struct Energies{T}
	K::Vector{T}
	Kx::Vector{T}
	Ky::Vector{T}
	Kz::Vector{T}
	A::Vector{T}  # показатель анизотропии
	Ex::Vector{T}
	By::Vector{T}
	Bz::Vector{T}

	function Energies{T}(N::Int) where {T<:Real}
		arrays = [zeros(T,N) for i in 1:fieldcount(Energies)]
		new(arrays...)
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
	Jx::Matrix{T}
	Jy::Matrix{T}
	Jz::Matrix{T}

	function Fields{T}(M, N) where {T<:Real}
		arrays = [zeros(T,M,N) for i in 1:fieldcount(Fields)]
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

function simulation!(e, i, ρ, φ, Ex, B, A, g, time, dir)
	field = Fields{Float64}(g.N, length(time))
	energy = Energies{Float64}(length(time))
	Ki = zeros(length(time))
	dt = step(time)
	@showprogress 1 "Computing..." for t in eachindex(time)
		# sources!(ρ, A, e, i, g)
		collect_sources!(ρ, A, g, [e, i])
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
		Ex .*= -1
		vector_potential!(A, g)
		curl!(B, A, g)
		leap_frog!(e, dt, Ex, B, A, g)
		leap_frog!(i, dt, Ex, B, A, g)
		# сбор данных
		Kx, Ky, Kz = kinetic_energy(e, A, g)
		energy.K[t] = Kx+Ky+Kz
		energy.Kx[t] = Kx
		energy.Ky[t] = Ky
		energy.Kz[t] = Kz
		energy.A[t] = 2Kz/(Kx+Ky) - 1
		Ki[t] = sum(kinetic_energy(i, A, g))
		energy.Ex[t] = field_energy(Ex, g)
		energy.By[t] = field_energy(B.y, g)
		energy.Bz[t] = field_energy(B.z, g)
		field.rho[:,t] .= ρ[g.in]
		field.phi[:,t] .= φ[g.in]
		field.Ay[:,t] .= A.y[g.in]
		field.Az[:,t] .= A.z[g.in]
		field.Ex[:,t] .= Ex[g.in]
		field.By[:,t] .= B.y[g.in]
		field.Bz[:,t] .= B.z[g.in]
	end
	
	write_SoA(dir*"energies.h5", energy)
	write_SoA(dir*"fields.h5", field)
	return
end


function simulation!(e, ::Nothing, ρ, φ, Ex, B, A, g, time, dir)
	Jx = similar(ρ)
	Jy = similar(ρ)
	Jz = similar(ρ)
	field = Fields{Float64}(g.N, length(time))
	energy = Energies{Float64}(length(time))
	Ki = zeros(length(time))
	dt = step(time)
	@showprogress 1 "Computing..." for t in eachindex(time)
		collect_sources!(ρ, A, g, [e])
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
		Ex .*= -1
		vector_potential!(A, g)
		curl!(B, A, g)
		current_densities!(Jx, Jy, Jz, A, e, g)
		leap_frog!(e, dt, Ex, B, A, g)
		# сбор данных
		Kx, Ky, Kz = kinetic_energy(e, A, g)
		energy.K[t] = Kx+Ky+Kz
		energy.Kx[t] = Kx
		energy.Ky[t] = Ky
		energy.Kz[t] = Kz
		energy.A[t] = 2Kz/(Kx+Ky) - 1
		energy.Ex[t] = field_energy(Ex, g)
		energy.By[t] = field_energy(B.y, g)
		energy.Bz[t] = field_energy(B.z, g)
		# fields_time[t] = t
		# fields_Jy[:,t] .= A.fy[g.in] .- A.μ[g.in].*A.y[g.in]
		# fields_Jz[:,t] .= A.fz[g.in] .- A.μ[g.in].*A.z[g.in]
		field.Jx[:,t] .= Jx[g.in]
		field.Jy[:,t] .= Jy[g.in]
		field.Jz[:,t] .= Jz[g.in]
		field.rho[:,t] .= ρ[g.in]
		field.phi[:,t] .= φ[g.in]
		field.Ay[:,t] .= A.y[g.in]
		field.Az[:,t] .= A.z[g.in]
		field.Ex[:,t] .= Ex[g.in]
		field.By[:,t] .= B.y[g.in]
		field.Bz[:,t] .= B.z[g.in]
	end
	
	write_SoA(dir*"energies.h5", energy)
	write_SoA(dir*"fields.h5", field)
	return
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

	num_model = NumericalModel(g)
	model.ion_bg || collect_sources!(num_model.ρ, num_model.A, num_model.g, [e, i])
	model.ion_bg && collect_sources!(num_model.ρ, num_model.A, num_model.g, [e])
	scalar_potential!(num_model.φ, num_model.ρ, num_model.g)
	gradient!(num_model.Ex, num_model.φ, num_model.g)
	num_model.Ex .*= -1
	vector_potential!(num_model.A, num_model.g)
	num_model.B.y .= 0
	num_model.B.z .= 0

	leap_frog_halfstep!(e, step(model.time), num_model.Ex, num_model.g)
	leap_frog_halfstep!(i, step(model.time), num_model.Ex, num_model.g)
	prestart!(e, i, num_model.ρ, num_model.φ, num_model.Ex, num_model.B, num_model.A, num_model.g, model.time[1:model.prestart_steps],
		min(model.uˣ, model.uʸ, model.uᶻ), (model.uˣ, model.uʸ, model.uᶻ))
	simulation!(e, i, num_model.ρ, num_model.φ, num_model.Ex, num_model.B, num_model.A, num_model.g, model.time, dir)
end
