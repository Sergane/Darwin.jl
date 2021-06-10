using TOML

struct Configuration
    data::Dict

    function Configuration(filename::String)
        new(TOML.parsefile(filename))
    end
end

function dict_to_str(data::Dict, indent_lvl=0, result="")
    indent_str = repeat(' '^2, indent_lvl) # здесь задается размер отступов (^2)
    for item_key in keys(data)
        item = data[item_key]
        item_str = replace(item_key, '_'=>' ')
        result *= indent_str * item_str * ":  "
        if isa(item, Dict)
            result *= '\n'
            result *= dict_to_str(item, indent_lvl+1)
        elseif isa(item, Array)
            result *= array_to_str(item, indent_lvl)
        else
            result *= repr(item) * '\n'
        end
    end
    return result
end

function array_to_str(data::Array, indent_lvl, result="")
    for item in data
        if isa(item, Dict)
            result *= '\n'
            result *= dict_to_str(item, indent_lvl+1)
        else
            result *= repr(item) * ",  "
        end
    end
    if !isa(last(data), Dict)
        pred = (a->a in ";, ")
        result = rstrip(pred, result)
        result *= '\n'
    end
    return result
end

# rstrip здесь убирает лишний перенос строки в конце
Base.show(io::IO, config::Configuration) = print(io, rstrip(dict_to_str(config.data),'\n'))

struct ParticleParameters
    name::String
    charge::Float64
    mass::Float64
    ppc::Int
    V_distribution::String
    V_params::NTuple{3,Float64}
    R_distribution::String
end

function ParticleParameters(ind::Int, species, ppc)
    name = species["name"][ind]
    mass = species["mass"][ind]
    charge = species["charge"][ind]
    V_distr  = species["V_distribution"][ind]["type"]
    V_params = Tuple(a for a in species["V_distribution"][ind]["params"])
    R_distr  = species["R_distribution"][ind]["type"]
    ParticleParameters(name, charge, mass, ppc, V_distr, V_params, R_distr)
end

struct NumericalParameters
	L::Float64
	cells_num::Int
    T::Float64
    dt::Float64
	prestart_steps::Int
    species::Vector{ParticleParameters}
end

function NumericalParameters(numerical, domain_size, species)
    L = domain_size["space"]
    T = domain_size["time"]

    cells_num = numerical["cells_num"]
    time_step = numerical["time_step"]
    prestart_steps = numerical["prestart_steps"]

    ppc_vector = numerical["particles_per_cell"]
    partical_params = [ParticleParameters(i, species, ppc) for (i,ppc) in enumerate(ppc_vector)]

    NumericalParameters(L, cells_num, T, time_step, prestart_steps, partical_params)
end

function NumericalParameters(config::Configuration)
    NumericalParameters(config.data["numerical"],
        config.data["physical"]["domain_size"],
        config.data["physical"]["species"])
end

function Base.show(io::IO, params::NumericalParameters)
    "NumericalParameters(\n" *
    "  T: $(params.T),\n" *
    "  dt: $(params.dt),\n" *
    "  L: $(params.L),\n" *
    "  cells num: $(params.cells_num),\n" *
    "  prestart steps: $(params.prestart_steps)\n" *
    ")" |> print
end