from lead_agent import GeneralAnalystAgent

# specialist_agents=False keeps this to a single LLM call (just the perspective)
agent = GeneralAnalystAgent(use_specialist_agents=False, use_perspective=True)

data = agent.gather_data("MU", perspective="serenity-aleabitoreddit")
assert data["perspective"] is not None, "Perspective didn't load — check the cache key / distiller output"

base  = agent.calculate_scores(data)                          # no weight  -> 3-factor blend
withp = agent.calculate_scores(data, perspective_weight=0.15) # with weight -> post-blend

print(f"perspective: {data['perspective']['score']}/100  ({data['perspective']['verdict']})")
print(f"overall WITHOUT perspective: {base['overall']}")
print(f"overall WITH perspective:    {withp['overall']}")
print(f"confidence: {withp['confidence']}")   # proves the KeyError fix

assert base["overall"] != withp["overall"], "Score didn't move — blend not wired"
print("✓ perspective loads, blends, and moves the score")