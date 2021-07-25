include("configuration.jl")
include("numericalmodel.jl")
include("datamodel.jl")

using HDF5
using LinearAlgebra: diagm
using ProgressMeter

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

	φ.ρ .-= mean_charge(species)  # ионный фон

	boundary_condition!(φ.ρ)
	boundary_condition!(A.μ)
	boundary_condition!(A.fy)
	boundary_condition!(A.fz)
	# interpolation_bc!(φ.ρ)
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
	q, m = particle.q, particle.m
	β = q / (m * PPC)

	ρ .*= q / PPC
	μ .= ρ .* (q/m)
	fy .*= β
	fz .*= β
end

@inline function mean_charge(species)
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

##

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

##

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


function simulation!(species, model, time, dir, field, energy)
	g = model.g
	φ = model.φ
	A = model.A
	Ex= model.Ex
	B = model.B
	Jx = similar(g.range)
	Jy = similar(Jx)
	Jz = similar(Jx)
	dt = step(time)

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
end


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
	# vector_potential!(model.A, model.g)
	# model.B.y .= 0
	# model.B.z .= 0

	for s in model.species
		leap_frog_halfstep!(s, step(model.time), model.Ex, model.g)
	end

	# u = min(uˣ,uʸ,uᶻ)
	# for s in species
	# 	s.px .*= u/uˣ
	# 	s.py .*= u/uʸ
	# 	s.pz .*= u/uᶻ
	# end
	# prestart!(model.species, model, model.time[1:params.prestart_steps])
	# for s in species
	# 	s.px .*= uˣ/u
	# 	s.py .*= uʸ/u
	# 	s.pz .*= uᶻ/u
	# end
	# J = [Densities{Float64}(g.N, length(time)) for i in eachindex(species)]
	field = Fields{Float64}(model.g[model.g.in], model.g.N, length(model.time))
	energy = Energies{Float64}(model.time, length(model.species))

	simulation!(model.species, model, model.time, dir, field, energy)
	
	write_SoA(dir*"energies.h5", energy)
	write_SoA(dir*"fields.h5", field)
	for (i,s) in enumerate(model.species)
		write_SoA(dir*"kinetic_energies_"*s.name*".h5", energy.K[i])
		# write_SoA(dir*"J_"*string(i)*".h5", J[i])
	end
end
