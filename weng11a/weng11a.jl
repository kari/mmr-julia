"""
weng11a
A Bayesian Approximation Method for Online Ranking
Algorithm 1
"""

using Distributions

struct Player
    "rating estimate ´𝜇´, Default is 25"
    rating::Float64
    "rating deviation ´σ2´, Default is (25/3)^2"
    rd::Float64
    "player's 'true' skill for simulation purposes, can be sampled from ´N(rating, RD)´"
    skill::Float64
end

Player() = Player(25, (25/3)^2, rand(Normal(25, (25/3)^2)))
Player(rating, rd) = Player(rating, rd, rand(Normal(rating, rd)))

"""
    update(teams, ranks, β², κ)

Return updated ´teams´'s ratings against each other with ranks ´ranks´ and the system parameters ´β²´ and ´κ´. 
"""
function update(teams::Array{Array{Player, 1}, 1}, ranks::Array{<:Number, 1}, β² = (25/6)^2, κ = 0.0001)
    k = length(teams)

    # Step 2
    𝜇 = [sum([j.rating for j in i]) for i in teams]
    σ² = [sum([j.rd for j in i]) for i in teams]

    new_teams = Array{Array{Player,1},1}(undef, 2)
    # Step 3
    for i in 1:k
        # Step 3.1
        Ωᵢ = 0
        ∆ᵢ = 0

        𝜇ᵢ = 𝜇[i]
        σ²ᵢ = σ²[i]
        σᵢ = sqrt(σ²ᵢ)
        # Step 3.1.1
        for q in [q for q in 1:k if q != i]
            𝜇_q = 𝜇[q]
            σ²_q = σ²[q]
    
            s = if ranks[q] > ranks[i]
                    1.0
                elseif ranks[q] < ranks[i]
                    0.0
                else
                    0.5
                end

            c_iq = sqrt(σ²ᵢ + σ²_q + 2*β²)
            c_qi = c_iq
            p̂_iq = exp(𝜇ᵢ/c_iq)/(exp(𝜇ᵢ/c_iq) + exp(𝜇_q/c_iq))
            p̂_qi = exp(𝜇_q/c_qi)/(exp(𝜇_q/c_qi) + exp(𝜇ᵢ/c_qi)) #p̂_qi = 1-p̂_iq

            γ_q = σᵢ/c_iq

            δ_q = σ²ᵢ / c_iq * (s - p̂_iq)
            η_q = γ_q * (σᵢ/c_iq)^2 * p̂_iq * p̂_qi

            # Step 3.1.2
            Ωᵢ += δ_q
            ∆ᵢ += η_q
        end

        # Step 3.2
        nᵢ = length(teams[i])
        new_teams[i] = Array{Player}(undef, nᵢ)
        for j in 1:nᵢ
            player = teams[i][j]
            𝜇_ij = player.rating
            σ²_ij = player.rd

            𝜇_ij = 𝜇_ij + σ²_ij/σ²ᵢ * Ωᵢ
            σ²_ij = σ²_ij * max(1 - σ²_ij/σ²ᵢ * ∆ᵢ, κ)

            new_teams[i][j] = Player(𝜇_ij, σ²_ij, player.skill)
        end
    end

    return new_teams

end

