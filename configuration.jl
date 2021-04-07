# export  Configuration

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

## test
config = Configuration("$(@__DIR__)/config.toml")
println(config)
