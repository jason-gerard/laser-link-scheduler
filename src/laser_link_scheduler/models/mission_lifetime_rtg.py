import numpy as np 

def mission_lifetime_rtg(
        P0: float, 
        decay_constant: float, 
        P_min: float
    ) -> float:
    """ 
    Compute mission lifetime under an RTG power model with exponential decay. 
    The radioisotope thermoelectric generator (RTG) is modeled as an exponentially decaying power source: 
        P(t) = P0 * exp(-λ * t) 
        where P(t) is the available power at time t, 
        P0 is the initial power, and λ is the decay constant. 
    Mission lifetime L_m is defined as the duration for which P(t) remains >= P_min. 
    Parameters ---------- P0 : float Initial available power at t = 0. Units: power (e.g., W). decay_constant : float Exponential decay constant λ. Units: 1 / time (must be > 0 for finite decay). The time unit of the returned lifetime matches the inverse of this unit. P_min : float Minimum operational power threshold. Units: power (same as P0). Returns ------- float Mission lifetime L_m in the same time units as 1/λ. For λ > 0 and P0 >= P_min: L_m = (1/λ) * ln(P0 / P_min) If P0 < P_min, returns 0.0. Raises ------ ValueError If P0 <= 0, P_min <= 0, or decay_constant <= 0. Notes ----- This definition treats lifetime as the first time the RTG power drops below the operational threshold (continuous-time model, no margins, no other loads). Examples -------- >>> mission_lifetime_rtg(P0=300.0, decay_constant=0.03, P_min=200.0) 13.515503603605481 """ 
    
    if P0 <= 0: 
        raise ValueError("Initial power P0 must be greater than 0.") 
    if P_min <= 0: 
        raise ValueError("Minimum operational power P_min must be greater than 0.") 
    if decay_constant <= 0: 
        raise ValueError("Decay constant must be greater than 0 for finite decay.") 
    if P0 < P_min: 
        return 0.0 
    
    lifetime = (1 / decay_constant) * np.log(P0 / P_min) 
    return lifetime 

if __name__ == "__main__": 
    P0 = 300.0 # Initial power in Watts 
    decay_constant = 0.03 # Decay constant in 1/years 
    P_min = 200.0 # Minimum operational power in Watts 
    lifetime = mission_lifetime_rtg(P0, decay_constant, P_min) 
    print(f"Mission lifetime under RTG power model: {lifetime:.2f} years")