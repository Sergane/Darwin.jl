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
	time::Vector{T}
	K::Vector{KineticEnergies{T}}
	fields::FieldEnergies{T}

	function Energies{T}(time, species_num::Int) where {T<:Real}
		N = length(time)
		new(time,
			[KineticEnergies{T}(N) for i in 1:species_num],
			FieldEnergies{T}(N))
	end
end

struct Fields{T}
	grid::Vector{T}
	rho::Matrix{T}
	phi::Matrix{T}
	Ay::Matrix{T}  # векторный потенциал
	Az::Matrix{T}
	Ex::Matrix{T}
	By::Matrix{T}
	Bz::Matrix{T}

	function Fields{T}(grid, M, N) where {T<:Real}
		arrays = [zeros(T,M,N) for i in 2:fieldcount(Fields)]
		new(grid, arrays...)
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
	h5open(dir, "w", swmr=true) do file
		for prop_name in propertynames(obj)
			field = getfield(obj, prop_name)
			write(file, string(prop_name), field)
		end
	end
end

function write_SoA(dir, energies::Energies)
	obj = energies.fields
	h5open(dir, "w", swmr=true) do file
		for prop_name in propertynames(obj)
			field = getfield(obj, prop_name)
			write(file, string(prop_name), field)
		end
		write(file, "time", energies.time)
	end
end