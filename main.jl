include("transducer.jl")
include("graphs.jl")

using Printf
using GraphPlot
using Compose

import Cairo, Fontconfig


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

# println("States: $(T.Q)")
# println("Finals: $(T.F)")
# println("Transitions DELTA: $(T.δ)")
# println("Transitions LAMBDA: $(T.λ)")

# function main()
#     D = read_file(filename)
#     return construct(D)
# end

# main()

# G, node_labels, edge_labels = transducer_to_graph(T)

# println(T.δ)
# println("")

# println(ne(G))
# println(length(edge_labels))

# graph = gplot(G, nodelabel=node_labels, edgelabel=collect(edge_labels))
# graph |> PDF("$(filename).pdf")

println("˅˅˅ Results ˅˅˅")
println("Number of states: $(length(T.Q))")
println("Number of transitions: $(length(T.δ))")