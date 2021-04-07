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

## Параметры:

struct Model
	time::AbstractRange
	L::Float64
	Nc::UInt
	Npc::UInt
	uˣ::Float64
	uʸ::Float64
	uᶻ::Float64
	ion_bg::Bool
	init_method::String
	prestart_steps::UInt

	g::Grid
end

function Model(time,
	L,
	Nc,
	Npc,
	uˣ,
	uʸ,
	uᶻ,
	ion_bg,
	init_method,
	prestart_steps)
	
	Model(time,
		L,
		Nc,
		Npc,
		uˣ,
		uʸ,
		uᶻ,
		ion_bg,
		init_method,
		prestart_steps,
		Grid(L, Nc, Npc))
end


begin
	using CSV
	using DataFrames

	model = Model(0:0.25:500,
		5.24,
		256,
		1000,
		0.0316,
		0.0316,
		0.1,
		true,
		"rand",
		200)
	println("A: $((model.uᶻ/model.uˣ)^2-1)")
	
	dir = isempty(ARGS) ? "test/" : ARGS[1]*"/"
	mkpath(dir)
	println("writing to ", abspath(dir))

	# TODO:
	# разбить модель на независимые абстракции
	# вывести здесь параметры модели 
end

#step(time)*√(uˣ^2+uʸ^2+uᶻ^2) ≤ L/2Nc

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

function init_data(s)
	data = DataFrame()
	data.x = s.x
	data.Vx = s.px
	data.Vy = s.py
	data.Vz = s.pz
	data
end

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

function particle_set(q, m, N, space, data, range)
	(; (space[k]=>view(data,range[k],:) for k in eachindex(space))...,
		:q=>q, :m=>m, :N=>N)
end

begin
	using Distributions

	N = model.g.Npc*model.g.N

	space = (:x, :px, :py, :pz)
	data = zeros((1+!model.ion_bg)*length(space), N)

	e = particle_set(-1, 1, N, space, data, 1:4)
	eval(Symbol("init_"*model.init_method*'!'))(e, model.L, (model.uˣ,model.uʸ,model.uᶻ)./√2..., (2,3,7,5))
	CSV.write(dir*"init_electron.csv", init_data(e))
	e.px .*= e.m
	e.py .*= e.m
	e.pz .*= e.m

	i = nothing
if !model.ion_bg
	i = particle_set(1, 1836, N, space, data, 5:8)
	K = √(e.m / i.m)
	eval(Symbol("init_"*model.init_method*'!'))(i, model.L, (model.uˣ,model.uʸ,model.uᶻ).*(K/√2)..., (2,3,7,5))
	CSV.write(dir*"init_ion.csv", init_data(i))
	i.px .*= i.m
	i.py .*= i.m
	i.pz .*= i.m
end
end;

function sources!(ρ, μ, f, e, i, g)
	ρ .= 0
	μ .= 0
	f .= 0
	for s in (e, i)
		m = s.m
		q = s.q
		@inbounds for k in 1:s.N
			j, l = g(s.x[k])
			r = 1 - l
			l /= g.Npc
			r /= g.Npc
			ρ[j]   += l*q
			ρ[j+1] += r*q
			μ[j]   += l*q^2/m
			μ[j+1] += r*q^2/m
			f[2,j] 	 += l*q/m*s.py[k]
			f[2,j+1] += r*q/m*s.py[k]
			f[3,j]   += l*q/m*s.pz[k]
			f[3,j+1] += r*q/m*s.pz[k]
		end # хранить сетки рядом?
	end
	boundary_condition!(ρ)
	boundary_condition!(μ)
	boundary_condition!(f)
	interpolation_bc!(ρ)
end

function sources!(ρ, μ, f, e, g)
	ρ .= 0
	μ .= 0
	f .= 0
	m = e.m
	q = e.q
	@inbounds for k in 1:e.N
		j, l = g(e.x[k])
		r = 1 - l
		l /= g.Npc
		r /= g.Npc
		ρ[j]   += l
		ρ[j+1] += r
		f[2,j] 	 += l*e.py[k]
		f[2,j+1] += r*e.py[k]
		f[3,j]   += l*e.pz[k]
		f[3,j+1] += r*e.pz[k]
	end
	ρ .*= q
	μ .= ρ.*(q/m)
	f .*= q/m
	boundary_condition!(ρ)
	ρ .-= e.q  # ионный фон
	boundary_condition!(μ)
	boundary_condition!(f)
	interpolation_bc!(ρ)
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

function vector_potential!(A, μ, f, g)
	N = g.N
	h² = g.h^2
    M = diagm(0=>-h².*μ[1:N].-2, 1=>ones(N-1), -1=>ones(N-1))
	M[N,1] = M[1,N] = 1
    A[2,1:N] .= M \ (-h²*f[2,1:N])
    A[3,1:N] .= M \ (-h²*f[3,1:N])
	interpolation_bc!(A,2)
	interpolation_bc!(A,3)
end

function curl!(B, A, g)
    for k in g.in
        B[2,k] = -(A[3,k+1]-A[3,k-1]) / 2g.h
        B[3,k] =  (A[2,k+1]-A[2,k-1]) / 2g.h
    end
    interpolation_bc!(B,2)
    interpolation_bc!(B,3)
end

## Сетки для источников и полей:

begin
	ρ = OffsetVector(zeros(model.g.N+2), 0:model.g.N+1)
	μ = similar(ρ)
	φ = similar(ρ)
	Ex = similar(ρ)
	f = OffsetMatrix(zeros(2,model.g.N+2), 2:3, 0:model.g.N+1)
	A = similar(f)
	B = similar(f)
	model.ion_bg || sources!(ρ, μ, f, e, i, model.g)
	model.ion_bg && sources!(ρ, μ, f, e, model.g)
	scalar_potential!(φ, ρ, model.g)
	gradient!(Ex, φ, model.g)
	Ex *= -1
	vector_potential!(A, μ, f, model.g)
	B .= 0
end;

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
		Ky += (s.py[j] - s.q*(l*A[2,i]+r*A[2,i+1]))^2
		Kz += (s.pz[j] - s.q*(l*A[3,i]+r*A[3,i+1]))^2
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
		Jy[j] 	+= l*(e.py[k] - e.q*(l*A[2,j]+r*A[2,j+1]))
		Jy[j+1] += r*(e.py[k] - e.q*(l*A[2,j]+r*A[2,j+1]))
		Jz[j]   += l*(e.pz[k] - e.q*(l*A[3,j]+r*A[3,j+1]))
		Jz[j+1] += r*(e.pz[k] - e.q*(l*A[3,j]+r*A[3,j+1]))
	end
	Jx .*= e.q/(e.m*g.Npc)
	Jy .*= e.q/(e.m*g.Npc)
	Jz .*= e.q/(e.m*g.Npc)
	boundary_condition!(Jx)
	boundary_condition!(Jy)
	boundary_condition!(Jz)
end

begin
	energy = DataFrame()
	energy.Ke = zeros(length(model.time))
	energy.Kex = zeros(length(model.time))
	energy.Key = zeros(length(model.time))
	energy.Kez = zeros(length(model.time))
	energy.A  = zeros(length(model.time))
	energy.Ki = zeros(length(model.time))
	energy.Ex = zeros(length(model.time))
	energy.By = zeros(length(model.time))
	energy.Bz = zeros(length(model.time))
end;


@inline it(E, l, j) = (l*E[j]+(1-l)*E[j+1])
@inline it(A, l, i, j) = (l*A[i,j]+(1-l)*A[i,j+1])


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
			 (s.py[k]-q*it(A,l,2,j))*it(B,l,3,j) -
			 (s.pz[k]-q*it(A,l,3,j))*it(B,l,2,j))
		s.x[k] += dt*s.px[k]/m
	end
	boundary_condition!(s, g)
end


function prestart!(e, i, ρ, μ, f, φ, Ex, B, A, g, time, u, (uˣ,uʸ,uᶻ))
	dt = step(time)
	e.px .*= u/uˣ
	e.py .*= u/uʸ
	e.pz .*= u/uᶻ
	i.px .*= u/uˣ
	i.py .*= u/uʸ
	i.pz .*= u/uᶻ
	@showprogress 1 "Prestart... " for t in time
		sources!(ρ, μ, f, e, i, g)
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
		Ex .*= -1
		vector_potential!(A, μ, f, g)
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


function prestart!(e, ::Nothing, ρ, μ, f, φ, Ex, B, A, g, time, u, (uˣ,uʸ,uᶻ))
	dt = step(time)
	e.px .*= u/uˣ
	e.py .*= u/uʸ
	e.pz .*= u/uᶻ
	@showprogress 1 "Prestart... " for t in time
		sources!(ρ, μ, f, e, g)
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
		Ex .*= -1
		vector_potential!(A, μ, f, g)
		curl!(B, A, g)
		leap_frog!(e, dt, Ex, B, A, g)
	end
	e.px .*= uˣ/u
	e.py .*= uʸ/u
	e.pz .*= uᶻ/u
	return
end


function simulation!(e, i, ρ, μ, f, φ, Ex, B, A, g, time, energy)
	fields_rho = zeros(g.N, length(time))
	fields_phi = zeros(g.N, length(time))
	fields_Ay = zeros(g.N, length(time))
	fields_Az = zeros(g.N, length(time))
	fields_Ex = zeros(g.N, length(time))
	fields_By = zeros(g.N, length(time))
	fields_Bz = zeros(g.N, length(time))
	dt = step(time)
	@showprogress 1 "Computing..." for t in eachindex(time)
		sources!(ρ, μ, f, e, i, g)
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
		Ex .*= -1
		vector_potential!(A, μ, f, g)
		curl!(B, A, g)
		leap_frog!(e, dt, Ex, B, A, g)
		leap_frog!(i, dt, Ex, B, A, g)
		# сбор данных
		Kx, Ky, Kz = kinetic_energy(e, A, g)
		energy.Ke[t] = Kx+Ky+Kz
		energy.Kex[t] = Kx
		energy.Key[t] = Ky
		energy.Kez[t] = Kz
		energy.A[t] = 2Kz/(Kx+Ky) - 1
		energy.Ki[t] = sum(kinetic_energy(i, A, g))
		energy.Ex[t] = field_energy(Ex, g)
		energy.By[t] = field_energy(B, 2, g)
		energy.Bz[t] = field_energy(B, 3, g)
		fields_rho[:,t] .= ρ[g.in]
		fields_phi[:,t] .= φ[g.in]
		fields_Ay[:,t] .= A[2,g.in]
		fields_Az[:,t] .= A[3,g.in]
		fields_Ex[:,t] .= Ex[g.in]
		fields_By[:,t] .= B[2,g.in]
		fields_Bz[:,t] .= B[3,g.in]
	end
	
	CSV.write(dir*"energy.csv", energy)
	h5open(dir*"fields.h5", "w") do file
		write(file, "rho", fields_rho)
		write(file, "phi", fields_phi)
		write(file, "Ay", fields_Ay)
		write(file, "Az", fields_Az)
		write(file, "Ex", fields_Ex)
		write(file, "By", fields_By)
		write(file, "Bz", fields_Bz)
	end
	return
end


function simulation!(e, ::Nothing, ρ, μ, f, φ, Ex, B, A, g, time, energy)
	Jx = similar(ρ)
	Jy = similar(ρ)
	Jz = similar(ρ)
	fields_rho = zeros(g.N, length(time))
	fields_Jx = zeros(g.N, length(time))
	fields_Jy = zeros(g.N, length(time))
	fields_Jz = zeros(g.N, length(time))
	fields_phi = zeros(g.N, length(time))
	fields_Ay = zeros(g.N, length(time))
	fields_Az = zeros(g.N, length(time))
	fields_Ex = zeros(g.N, length(time))
	fields_By = zeros(g.N, length(time))
	fields_Bz = zeros(g.N, length(time))
	dt = step(time)
	@showprogress 1 "Computing..." for t in eachindex(time)
		sources!(ρ, μ, f, e, g)
		scalar_potential!(φ, ρ, g)
		gradient!(Ex, φ, g)
		Ex .*= -1
		vector_potential!(A, μ, f, g)
		curl!(B, A, g)
		current_densities!(Jx, Jy, Jz, A, e, g)
		leap_frog!(e, dt, Ex, B, A, g)
		# сбор данных
		Kx, Ky, Kz = kinetic_energy(e, A, g)
		energy.Ke[t] = Kx+Ky+Kz
		energy.Kex[t] = Kx
		energy.Key[t] = Ky
		energy.Kez[t] = Kz
		energy.A[t] = 2Kz/(Kx+Ky) - 1
		energy.Ex[t] = field_energy(Ex, g)
		energy.By[t] = field_energy(B, 2, g)
		energy.Bz[t] = field_energy(B, 3, g)
		# fields_time[t] = t
		# fields_Jy[:,t] .= f[2,g.in] .- μ[g.in].*A[2,g.in]
		# fields_Jz[:,t] .= f[3,g.in] .- μ[g.in].*A[3,g.in]
		fields_Jx[:,t] .= Jx[g.in]
		fields_Jy[:,t] .= Jy[g.in]
		fields_Jz[:,t] .= Jz[g.in]
		fields_rho[:,t] .= ρ[g.in]
		fields_phi[:,t] .= φ[g.in]
		fields_Ay[:,t] .= A[2,g.in]
		fields_Az[:,t] .= A[3,g.in]
		fields_Ex[:,t] .= Ex[g.in]
		fields_By[:,t] .= B[2,g.in]
		fields_Bz[:,t] .= B[3,g.in]
	end
	
	CSV.write(dir*"energy.csv", energy)
	h5open(dir*"fields.h5", "w") do file
		write(file, "rho", fields_rho)
		write(file, "Jx", fields_rho)
		write(file, "Jy", fields_rho)
		write(file, "Jz", fields_rho)
		write(file, "phi", fields_phi)
		write(file, "Ay", fields_Ay)
		write(file, "Az", fields_Az)
		write(file, "Ex", fields_Ex)
		write(file, "By", fields_By)
		write(file, "Bz", fields_Bz)
	end
	return
end

let
	leap_frog_halfstep!(e, step(model.time), Ex, model.g)
	leap_frog_halfstep!(i, step(model.time), Ex, model.g)
	prestart!(e, i, ρ, μ, f, φ, Ex, B, A, model.g, model.time[1:model.prestart_steps],
		min(model.uˣ, model.uʸ, model.uᶻ), (model.uˣ, model.uʸ, model.uᶻ))
	simulation!(e, i, ρ, μ, f, φ, Ex, B, A, model.g, model.time, energy)
end
