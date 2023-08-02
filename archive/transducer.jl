using StructEquality

import Base.Iterators 
import Base.==

const SIGMA = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', 
'1', '2', '3', '4', '5', '6', '7', '8', '9']


struct State
    index::Number
end

function Base.:(==)(state::State, other::State)
    state.index == other.index
end

function Base.isless(state::State, other::State)
    state.index < other.index
end

function Base.hash(state::State)
    state.index
end

@def_structequal struct Signature
    is_final::Bool
    output::String
    outgoing::Set{Tuple{Char, String, State}}
end

function Base.:(==)(signature::Signature, other::Signature)::Bool
    signature.is_final == other.is_final && signature.output == other.output && issetequal(signature.outgoing, other.outgoing)
end

function Base.hash(signature::Signature)::Number
    (2^hash(signature.outgoing)) * (3^hash(signature.output)) + (signature.is_final ? 1 : 0)
end


mutable struct Transducer
    Q::Set{State}
    s::State
    F::Set{State}
    δ::Dict{Tuple{State, Char}, State}
    λ::Dict{Tuple{State, Char}, String}
    ι::String
    ψ::Dict{State, String}
    itr::Dict{State, Number}

    delta_state_to_chars::Dict{State, Vector{Char}}
end


# function common_prefix(s1::String, s2::String)::String
#     if s1 == "" || s2 == "" || s1[begin] != s2[begin]
#         return ""
#     end

#     return s1[1:minimum(l for l in 1:length(s1)+1 if l >= length(s1) || l >= length(s2) || s1[l] != s2[l])]
# end

function common_prefix(s1::String, s2::String)::String
    n = min(length(s1), length(s2))
    for i in 1:n
        if s1[i] != s2[i]
            return s1[1:i-1]
        end
    end
    return s1[1:n]
end


function remainder_suffix(w::String, s::String)::String
    return s[length(w)+1 : end]
end

function calcSignature(T::Transducer, q::State)::Signature
    return Signature(
        in(q, T.F),
        in(q, T.F) ? T.ψ[q] : "",
        Set((c, T.λ[(q, c)], T.δ[(q, c)]) for c in T.delta_state_to_chars[q])
    )
end


function state_seq(δ::Dict{Tuple{State, Char}, State}, q::State, w::String)::Vector{State}
    path = [q]
    state = q
    @inbounds for a in w
        if haskey(δ, (state, a))
            state = δ[(state, a)]
            path = push!(path, state)
        else
            break
        end
    end
    return path
end


function reduce_except(T::Transducer, a::String, l::Number, h::Dict{Signature, State})::Tuple{Transducer, Dict{Signature, State}}
    path = state_seq(T.δ, T.s, a)

    @inbounds for i in 0:length(path)-l-2
        p = lastindex(path) - i
        # println("\tRunning calcSignature")
        # h_q = @time calcSignature(T, path[p])
        # println("\tRan calcSignature\n")

        h_q = calcSignature(T, path[p])

        
        # println("\tEntering the if")
        if !haskey(h, h_q)
            h[h_q] = path[p]
        else
            # println("Entered the else")

            delete!(T.Q, path[p])
            delete!(T.F, path[p])

            if haskey(T.delta_state_to_chars, path[p])

                @inbounds for s in T.delta_state_to_chars[path[p]]
                    delete!(T.δ, (path[p], s))
                    delete!(T.λ, (path[p], s))
                end
                delete!(T.delta_state_to_chars, path[p])
            end

            # delete!(T.δ, (path[p-1], a[p-1]))
            T.δ[(path[p-1], a[p-1])] = h[h_q]

            delete!(T.ψ, path[p])
        end
        # println("\tExited if\n")

    end

    # @assert Set(keys(T.δ)) == Set(keys(T.λ))
    # @assert Set(keys(T.ψ)) == T.F
    # @assert issubset(Set(map(x -> x.state, [x for x in keys(T.δ)])), T.Q)
    return T, h
end


function construct_from_first_entry(a::String, b::String)::Tuple{Transducer, Dict{Signature, State}, Number}
    states = [State(i) for i in 1:length(a) +1]

    Q = Set(states)
    s = states[begin]
    F = Set([states[end]])

    D = Dict(((states[i], a[i]), states[i+1]) for i in 1:lastindex(a))
    L = Dict(((states[i], a[i]), "") for i in 1:lastindex(a))
    I = b
    P = Dict(states[end] => "")
    itr = Dict((state, 1) for state in states)
    itr[s] = 0

    delta_state_to_chars::Dict{State, Vector{Char}} = Dict((states[i], [a[i]]) for i in 1:lastindex(a))
    delta_state_to_chars[states[lastindex(a) + 1]] = []


    return Transducer(Q, s, F, D, L, I, P, itr, delta_state_to_chars), Dict(), length(a) + 1
end

function add_new_entry(T::Transducer, h::Dict{Signature, State}, v::String, beta::String, u::String, state_counter::Number)::Tuple{Transducer, Dict{Signature, State}, Int32}
    t::Vector{State} = state_seq(T.δ, T.s, v)
    k::Number = lastindex(t) - 1

    # println("Running reduce_except")
    # T, h = @time reduce_except(T, u, k, h)
    # println("Ran reduce_except\n")
    T, h = reduce_except(T, u, k, h)

    new_states = [State(state_counter + i) for i in 1:length(v) - k]
    state_counter += length(v) - k

    t1 = append!(t, new_states)

    for n in new_states
        push!(T.Q, n)
        T.itr[n] = 0
    end

    push!(T.F, new_states[end])

    for i in k+1:lastindex(v)
        T.δ[(t1[i], v[i])] = t1[i+1]
        T.delta_state_to_chars[t1[i]] = haskey(T.delta_state_to_chars, t1[i]) ? push!(T.delta_state_to_chars[t1[i]], v[i]) : [v[i]]
        T.itr[t1[i+1]] += 1
    end
    T.delta_state_to_chars[new_states[end]] = []

    # build lambda and psi

    # println("Running push_output_forward")
    # T.λ, T.ψ = @time push_output_forward(T, t1, v, beta, k)
    # println("Ran reduce_except")
    T.λ, T.ψ = push_output_forward(T, t1, v, beta, k)
    T.ι = common_prefix(T.ι, beta)

    return T, h, state_counter
end


function push_output_forward(T::Transducer, t::Vector{State}, v::String, beta::String, k::Number)::Tuple{Dict{Tuple{State, Char}, String}, Dict{State, String}}
    println("pushing out $((v, beta))")

    L = T.λ
    P = T.ψ
    c = "" 
    l = ""
    b = beta
    
    for j in 0:k
        L_i = j == 0 ? T.ι : L[(t[j], v[j])]

        println("L_i: $(L_i)")
        println("L: $(L)")
        println("P: $(P)")
        println("c: $(c)")
        println("l: $(l)")
        println("b: $(b)")
        println()


        c = common_prefix(l * L_i, b)
        l = remainder_suffix(c, l * L_i)
        b = remainder_suffix(c, b)

        X_trans = Dict(((t[j+1], s), L[(t[j+1], s)]) for s in SIGMA if haskey(T.δ, (t[j+1], s)) && s != v[j+1])

        L_trans = L
        for (x_in, x_out) in X_trans
            L_trans[x_in] = l * x_out
        end
        # for s in T.delta_state_to_chars[t[j+1]] 
        #     if s != v[j+1]
        #     # L_trans[x_in] = l*x_out
        #         x_in = (t[j+1], s)
        #         x_out = L[(t[j+1], s)]
        #         L[x_in] = l * x_out
        #     end
        # end 

        L = L_trans
        if j != 0
            L[(t[j], v[j])] = c
        end

        if t[j+1] in T.F
            cacheP = T.ψ[t[j+1]]
            P[t[j+1]] = l * cacheP
        end
    end

    # println("L_i: $(L_i)")
    println("L: $(L)")
    println("P: $(P)")
    println("c: $(c)")
    println("l: $(l)")
    println("b: $(b)")
    println()

    T.λ = L
    T.λ[(t[k+1], v[k+1])] = b
    for r in k+2:length(v)
        T.λ[(t[r], v[r])] = ""
    end


    T.ψ = P
    T.ψ[t[end]] = ""

    return T.λ, T.ψ
end


function pretty_print(T, dict)
    print("[")
    for (i, x) in enumerate(dict)
        in = x[1]
        out = x[2]

        print("[\"$(in[1].index)\", \"$(in[2])\", \"$(out.index)\", \"$(T.λ[in])\"]")

        if i != length(dict)
            print(",")
        end
    end
    print("]")
end
     
function construct(D::Vector{Tuple{String, String}})::Tuple{Transducer, Dict{Signature, State}}

    T, h, state_counter = construct_from_first_entry(D[1][1], D[1][2])

    #pretty_print(T, sort(collect(T.δ), by=x->x[1]))

    for i in 2:length(D)

        i % 1000 == 0 && println("ENTRY $(i); WORD $(D[i][1])")
        # println("Adding ENTRY $(i) : $(D[i])")
        T, h, state_counter = add_new_entry(T, h, D[i][1], D[i][2], D[i-1][1], state_counter)

        #@assert Set(keys(T.δ)) == Set(keys(T.λ))
    end

    T, h = reduce_except(T, D[end][1], 0, h)

    return T, h
end

