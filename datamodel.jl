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