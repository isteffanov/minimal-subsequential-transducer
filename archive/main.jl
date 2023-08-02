include("transducer.jl")

using Printf
using Profile
using PProf


function read_file(filename::String)::Array{Tuple{String, String}}
    println("BEGINNING TO READ THE FILE")

    D::Array{Tuple{String,String}} = []

    f = open(filename, "r")
    for l in readlines(f)
        s = split(l, "\t", limit=2)
        x::String = s[1]
        y::String = s[2]

        push!(D, (x, y))
        # break
    end
    close(f)

    println("READ THE WHOLE FILE")

    return D
end

filename = ARGS[1]

D = read_file(filename)

T, h = construct(D)

# cast = () -> construct(D[begin:5000])

# cast()
# Profile.clear()
# @profile cast()
# pprof()



# println("States: $(T.Q)")
# println("Finals: $(T.F)")
# println("Transitions DELTA: $(T.δ)")
# println("Transitions LAMBDA: $(T.λ)")

# println("˅˅˅ Results ˅˅˅")
# println("Number of states: $(length(T.Q))")
# println("Number of transitions: $(length(T.δ))")