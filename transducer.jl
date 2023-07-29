using StructEquality
import Base.Iterators 

SIGMA = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 
    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', 
    '7', '8', '9']
state_counter = Iterators.countfrom(1, 1)

@def_structequal struct State
    index::Number
    function State()
        head, tail = Iterators.peel(state_counter)
        index = head
        global state_counter = tail
        new(index)
    end
end

@def_structequal struct TransDomain
    state::State
    label::Char
end

function Base.isequal(A::TransDomain, B::TransDomain)
    A.state == B.state && A.label == B.label
end

function Base.hash(A::TransDomain)
    (2^hash(A.state.index))*(3^hash(A.label))
end

@def_structequal struct Signature
    is_final::Bool
    output::String
    outgoing::Set{Tuple{Char, String, State}}
end

mutable struct Transducer
    Q::Set{State}
    s::State
    F::Set{State}
    δ::Dict{TransDomain, State}
    λ::Dict{TransDomain, String}
    ι::String
    ψ::Dict{State, String}
    itr::Dict{State, Number}
    min_except::String
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

# println(common_prefix("asd", "as"))
# println(common_prefix("as", "asd"))
# println(common_prefix("", "as"))
# println(common_prefix("as", ""))

function remainder_suffix(w::String, s::String)::String
    return s[length(w)+1 : end]
end

# println(remainder_suffix("as", "asdf"))
# println(remainder_suffix("as", "adf"))
# println(remainder_suffix("", "asdf"))
# println(remainder_suffix("asdf", "asdf"))

function calcSignature(T::Transducer, q::State)::Signature
    return Signature(
        q in T.F,
        q in T.F ? T.ψ[q] : "",
        Set(map(c -> (c, T.λ[TransDomain(q, c)], T.δ[TransDomain(q, c)]), filter(c -> haskey(T.δ, TransDomain(q, c)), SIGMA)))
    )
end


function state_seq(δ::Dict{TransDomain, State}, q::State, w::String)::Array{State}
    path = [q]
    state = q
    for a in w
        if haskey(δ, TransDomain(state, a))
            state = δ[TransDomain(state, a)]
            path = push!(path, state)
        else
            break
        end
    end
    return path
end


function reduce_except(T::Transducer, a::String, l::Number, h::Dict{Signature, State})::Tuple{Transducer, Dict{Signature, State}}
    path = state_seq(T.δ, T.s, a)

    for i in 0:length(path)-l-1
        p = length(path) - i
        h_q = calcSignature(T, path[p])
        
        if !haskey(h, h_q)
            h[h_q] = path[p]
        else
            # println(path[p], " ", path[p-1])
            # println(typeof(path[p]))

            delete!(T.Q, path[p])
            delete!(T.F, path[p])

            for s in SIGMA
                # println(s)
                delete!(T.δ, TransDomain(path[p], s))
            end
            delete!(T.δ, TransDomain(path[p-1], a[p-1]))
            T.δ[TransDomain(path[p-1], a[p-1])] = h[h_q]

            for s in SIGMA
                delete!(T.λ, TransDomain(path[p], s))
            end

            delete!(T.ψ, path[p])
        end

        # println("States: $(T.Q)")
        # println("Transitions: $(T.δ)")
        # println("")
    end

    @assert Set(keys(T.δ)) == Set(keys(T.λ))
    @assert Set(keys(T.ψ)) == T.F
    @assert issubset(Set(map(x -> x.state, [x for x in keys(T.δ)])), T.Q)
    return T, h
end


function construct_from_first_entry(a::String, b::String)::Tuple{Transducer, Dict{Signature, State}}
    states = [State() for _ in 1:length(a) +1]

    Q = Set(states)
    s = states[begin]
    F = Set([states[end]])

    D = Dict([(TransDomain(states[i], a[i]), states[i+1]) for i in 1:lastindex(a)])
    L = Dict([(TransDomain(states[i], a[i]), "") for i in 1:lastindex(a)])
    I = b
    P = Dict([(states[end], "")])
    itr = Dict([(state, 1) for state in states[2:length(a) + 1]])
    itr[s] = 0

    return Transducer(Q, s, F, D, L, I, P, itr, a), Dict()
end

function add_new_entry(T::Transducer, h::Dict{Signature, State}, v::String, beta::String, u::String)::Tuple{Transducer, Dict{Signature, State}}
    t = state_seq(T.δ, T.s, v)
    println("Path for $(v): $(t)")
    k = length(t) - 1

    T, h = reduce_except(T, u, k, h)

    # println(t)
    # println(k)
    # println(v)
    # println("-------------")

    new_states = [State() for _ in 1:length(v) - k]
    t1 = append!(t, new_states)

    for n in new_states
        push!(T.Q, n)
        T.itr[n] = 0
    end

    push!(T.F, new_states[end])

    for i in k+1:length(v)
        T.δ[TransDomain(t1[i], v[i])] = t1[i+1]
        T.itr[t1[i+1]] += 1
    end

    # build lambda and psi

    println("Adding ($(v), $(beta))")

    L = T.λ
    P = T.ψ
    c = ""
    l = ""
    b = beta
    
    for j in 0:k

        L_i = j == 0 ? T.ι : L[TransDomain(t[j], v[j])]
        
        j != 0 && println("(t_$(j), v_$(j)): ($(t[j]), $(v[j]))")
        # println("L: $(L)")
        # println("P: $(P)")
        # println("c: $(c)")
        # println("l: $(l)")
        # println("b: $(b)")
        # println("L_$(j): $(L_i)")
        # println("")
        

        c = common_prefix(l * L_i, b)
        l = remainder_suffix(c, l * L_i)
        b = remainder_suffix(c, b)

        # X_trans = Dict([(TransDomain(t[j], s), L[(t[j], s)]) for s in SIGMA if s != v[j] && haskey(T.δ, TransDomain(t[j], s))])

        X_trans = Dict(
            map(
                s -> (TransDomain(t[j+1], s), L[TransDomain(t[j+1], s)]),
                filter(
                    s -> s != v[j+1] && haskey(T.δ, TransDomain(t[j+1], s)),
                    SIGMA
                )
            )
        )

        
        L_trans = L
        for x in keys(X_trans)
            delete!(L_trans, x)
        end
        for (x_in, x_out) in X_trans
            L_trans[x_in] = l*x_out
        end 

        if j == 0
            L = L_trans
        else
            L = delete!(L_trans, TransDomain(t[j], v[j]))
            L[TransDomain(t[j], v[j])] = c
        end

        if t[j+1] in T.F
            cacheP = T.ψ[t[j+1]]
            delete!(P, t[j+1])
            P[t[j+1]] = l * cacheP
        end
    end

    # println("L: $(L)")
    # println("P: $(P)")
    # println("c: $(c)")
    # println("l: $(l)")
    # println("b: $(b)")
    # println("")

    T.ι = common_prefix(T.ι, beta)

    T.λ = L
    T.λ[TransDomain(t[k+1], v[k+1])] = b
    for r in k+2:length(v)
        T.λ[TransDomain(t1[r], v[r])] = ""
    end


    T.ψ = P
    T.ψ[t1[end]] = ""

    return T, h
end
     
function construct(D::Array{Tuple{String, String}})::Tuple{Transducer, Dict{Signature, State}}

    T, h = construct_from_first_entry(D[1][1], D[1][2])

    println("States: $(T.Q)")
    println("Finals: $(T.F)")
    println("Transitions DELTA: $(T.δ)")
    println("Transitions LAMBDA: $(T.λ)")
    println("")

    for i in 2:length(D)

        i % 100000 == 0 && println("ENTRY $(i); WORD $(D[i][1])")

        T, h = add_new_entry(T, h, D[i][1], D[i][2], D[i-1][1])

        println("States: $(T.Q)")
        println("Finals: $(T.F)")
        println("Transitions DELTA: $(T.δ)")
        println("Transitions LAMBDA: $(T.λ)")
        println("")
        #@assert Set(keys(T.δ)) == Set(keys(T.λ))
    end

    T, h = reduce_except(T, D[end][1], 0, h)


    println("States: $(T.Q)")
    println("Finals: $(T.F)")
    println("Transitions DELTA: $(T.δ)")
    println("Transitions LAMBDA: $(T.λ)")
    println("")

    return T, h
end