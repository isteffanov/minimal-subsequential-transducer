include("transducer.jl")

using Graphs

function transducer_to_graph(T::Transducer)::Tuple{DiGraph, Array{Number}, Vector{String}}

    G = DiGraph(length(T.Q))
    edge_labels::Vector{String} = []

    println("Number of vertecies: $(nv(G))")
    println("Number of states: $(length(T.Q))")

    for (trans, to) in T.δ
        from = trans.state
        d_label = trans.label      

        push!(edge_labels, "$(d_label):$(T.λ[TransDomain(from, d_label)])")
        if !(add_edge!(G, from.index, to.index))
            # println("ne stana :( - $(from.index) kum $(to.index)")

        else
            # println("stana :) - $(from.index) kum $(to.index)")
        end
    end

    return G, [x.index for x in T.Q], edge_labels
    
end

