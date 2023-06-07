"""
Glicko-2 implementation in Julia
For reference see http://www.glicko.net/glicko/glicko2.pdf
"""

using Distributions

struct Player
    "rating estimate ´r´, Glicko default is 1500"
    rating::Float64
    "rating deviation ´RD´, Glicko default is 350"
    rd::Float64
    "player volatility ´σ´, Glicko 'default' is 0.06"
    volatility::Float64
    "player's 'true' skill for simulation purposes, can be sampled from ´N(r, RD)´"
    skill::Float64
end

Player() = Player(1500, 350, 0.06, rand(Normal(1500, 350)))
Player(rating, rd) = Player(rating, rd, 0.06, rand(Normal(rating, rd)))
Player(rating, rd, skill) = Player(rating, rd, 0.06, skill)

"""
    update(player, opponents, s, 𝜏)

Return updated ´player´'s ratings against ´opponents´ with outcomes ´s´ and the system parameter ´𝜏´. 

Glicko-2 paper recommends ´𝜏´ is in range 0.3 - 1.2 but does not offer a default.
"""
function update(player::Player, opponents::Array{Player, 1}, s::Array{<:Number, 1}, 𝜏::Float64)
    𝜇 = (player.rating - 1500)/173.7178
    𝜙 = player.rd/173.7178
    𝜎 = player.volatility
    𝜇ⱼ = map(o->(o.rating - 1500)/173.7178, opponents)
    𝜙ⱼ = map(o->o.rd/173.7178, opponents)

    g(𝜙) = 1/sqrt(1 + 3𝜙^2/π^2)
    E(𝜇, 𝜇ⱼ, 𝜙ⱼ) = 1/(1 + exp(-g(𝜙ⱼ)*(𝜇 - 𝜇ⱼ)))
    v = sum(@. g(𝜙ⱼ)^2 * E(𝜇, 𝜇ⱼ, 𝜙ⱼ) * (1 - E(𝜇, 𝜇ⱼ, 𝜙ⱼ)))^(-1)
    ∆ = v * sum(@. g(𝜙ⱼ) * (s - E(𝜇, 𝜇ⱼ, 𝜙ⱼ)))

    a = log(𝜎^2)
    f(x) = exp(x) * (∆^2 - 𝜙^2 - v - exp(x)) / 2(𝜙^2 + v + exp(x))^2 - (x - a)/𝜏^2
    ε = 0.000001

    A = a
    if ∆^2 > 𝜙^2 + v
        B = log(∆^2 - 𝜙^2 - v)
    else
        k = 1
        while f(a - k*𝜏) < 0
            k = k + 1
        end
        B = a - k*𝜏
    end
    f_A = f(A)
    f_B = f(B)
    while abs(B - A) > ε
        C = A + (A - B)*f_A/(f_B - f_A)
        f_C = f(C)
        if f_C * f_B < 0
            A = B
            f_A = f_B
        else
            f_A = f_A/2
        end
        B = C
        f_B = f_C
    end
    𝜎′ = exp(A/2)

    𝜙ˢ = sqrt(𝜙^2 + 𝜎′^2)
    𝜙′ = 1/sqrt(1/𝜙ˢ^2 + 1/v)
    𝜇′ = 𝜇 + 𝜙′^2 * sum(@. g(𝜙ⱼ) * (s - E(𝜇, 𝜇ⱼ, 𝜙ⱼ)))

    r′ = 173.7178𝜇′ + 1500
    RD′ = 173.7178𝜙′

    return Player(r′, RD′, 𝜎′, player.skill)
end

"""
    expected_outcome(player1, player2)

Return expected outcome of a match.

From original Glicko paper, http://www.glicko.net/glicko/glicko.pdf
"""
function expected_outcome(player1::Player, player2::Player)
    q = log(10)/400
    g(RD) = 1/sqrt(1 + 3q^2*RD^2/π^2)
    E(rᵢ, rⱼ, RDᵢ, RDⱼ) = 1/(1 + 10^(-g(sqrt(RDᵢ^2 + RDⱼ^2))*(rᵢ - rⱼ)/400))

    return E(player1.rating, player2.rating, player1.rd, player2.rd)
end

"""
    true_expected_outcome(player1, player2, s = 350)

Calculate expected outcome using "hidden" skill attribute. 

Uses ´Logistic´  distribution, other option would be to use ´Normal´. ´s´ = standard deviation of the scoring system.
"""
function true_expected_outcome(player1::Player, player2::Player, s = 350)
    return cdf(Logistic(0, s), player1.skill - player2.skill)
end

# Examples/Tests
@show update(Player(), [Player() for i = 1:10], rand(Bernoulli(), 10), 0.3) # FIXME: use Bernoulli p for each matchup from true_expected_outcome
@show update(Player(1500, 200), [Player(1400, 30), Player(1550, 100), Player(1700, 300)], [1, 0, 0], 0.5) # Example from Glicko-2 paper, should return Player(1464, 151.52, 0.05999)
@show expected_outcome(Player(1400, 80), Player(1500, 150)) # Example from Glicko paper, should return 0.376
@show expected_outcome(Player(1500, 350), Player(1500, 350)) # Should return 0.5
@show expected_outcome(Player(1400, 350), Player(1500, 350)) # Should return ~0.43
@show true_expected_outcome(Player(1500, 350, 1500), Player(1500, 350, 1500)) # Should return 0.5
@show true_expected_outcome(Player(1400, 80, 1400), Player(1500, 150, 1500)) # Should return ~0.42
