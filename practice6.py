"""Practice 6: Bayes, HMM filtering and two experiments. Python 3."""


def bayes(prior, likelihood_o, likelihood_c):
    score_o = prior * likelihood_o
    score_c = (1.0 - prior) * likelihood_c
    total = score_o + score_c
    if total == 0:
        raise ValueError("Observation has zero probability")
    return score_o / total


def hmm_filter(observations, prior=0.30):
    emission = {"BLOCKED": (0.80, 0.20),
                "CLEAR": (0.20, 0.80)}
    belief = prior
    rows = []
    for t, reading in enumerate(observations, 1):
        # The first reading updates the prior without a transition.
        predicted = (belief if t == 1 else
                     0.70 * belief + 0.20 * (1.0 - belief))
        lo, lc = emission[reading]
        score_o = predicted * lo
        score_c = (1.0 - predicted) * lc
        total = score_o + score_c
        belief = bayes(predicted, lo, lc)
        rows.append((t, reading, predicted, score_o,
                     score_c, total, belief, 1.0 - belief))
    return rows


def main():
    print("PART A")
    print(f"Single BLOCKED: {bayes(0.30, 0.80, 0.20):.6f}")
    print(f"Single CLEAR:   {bayes(0.30, 0.20, 0.80):.6f}")
    print("\nHMM: no transition before t1")
    print("t  reading   predict_O   score_O    score_C    "
          "total      post_O     post_C")
    observations = ["BLOCKED", "CLEAR", "BLOCKED", "BLOCKED"]
    for t, reading, pred, so, sc, total, po, pc in hmm_filter(observations):
        print(f"{t}  {reading:7}   {pred:.6f}   {so:.6f}   "
              f"{sc:.6f}   {total:.6f}   {po:.6f}   {pc:.6f}")
    print("\nD1: change prior, same sensor")
    for prior in (0.10, 0.30, 0.50, 0.70):
        print(f"prior={prior:.2f} -> P(O|BLOCKED)="
              f"{bayes(prior, 0.80, 0.20):.6f}")
    print("\nD2: change false alarm, prior=0.30")
    for false_alarm in (0.20, 0.40):
        # A complementary CLEAR probability preserves each CPT row.
        emission = {"BLOCKED": (0.80, false_alarm),
                    "CLEAR": (0.20, 1.0 - false_alarm)}
        lo, lc = emission["BLOCKED"]
        print(f"false_alarm={false_alarm:.2f}, "
              f"P(CLEAR|C)={emission['CLEAR'][1]:.2f} -> "
              f"P(O|BLOCKED)={bayes(0.30, lo, lc):.6f}")


if __name__ == "__main__":
    main()
