using Glicko2
using Test
using Distributions

# Examples/Tests
@show update(Player(), [Player() for i = 1:10], rand(Bernoulli(), 10), 0.3) # FIXME: use Bernoulli p for each matchup from true_expected_outcome

@show update(Player(1500, 200), [Player(1400, 30), Player(1550, 100), Player(1700, 300)], [1, 0, 0], 0.5) # Example from Glicko-2 paper, should return Player(1464, 151.52, 0.05999)

p = Player(1500, 200) # Iterative approach
p = update(p, Player(1400, 30), 1, 0.5)
p = update(p, Player(1550, 100), 0, 0.5)
p = update(p, Player(1700, 300), 0, 0.5)
@show p # Should return close to above, Player(1464, 151.87, 0.05999)

# Example from Glicko paper, should return 0.376
@test round(expected_outcome(Player(1400, 80), Player(1500, 150)), digits = 3) == 0.376
# Should return 0.5
@test round(expected_outcome(Player(1500, 350), Player(1500, 350)), digits = 1) == 0.5 
@test round(expected_outcome(Player(1400, 350), Player(1500, 350)), digits = 3) == 0.423 # Should return ~0.423

# Should return 0.5
@test round(true_expected_outcome(Player(1500, 350, 1500), Player(1500, 350, 1500)), digits = 1) == 0.5
# Should return ~0.43
@test round(true_expected_outcome(Player(1400, 80, 1400), Player(1500, 150, 1500)), digits = 3) == 0.429

# Should return (1441, 1559)
@test round.(ci(Player(1500, 30))) == (1441, 1559) 
