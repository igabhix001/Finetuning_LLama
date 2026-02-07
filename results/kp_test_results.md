# KP Astrology Model Test Results

**Date:** 2026-02-07 06:27:41  
**Model:** kp-astrology-llama (Llama 3.1 8B fine-tuned)  
**Chart:** TestUser, 01.01.1990, 10:00, Aquarius Lagna  
**Temperature:** 0.4  
**RAG:** Enabled (top-k=5)  

## Summary

| # | ID | Category | Weight | Latency | RAG | Rule Cited | Confidence | Score |
|---|-----|----------|--------|---------|-----|------------|------------|-------|
| 1 | M1 | Marriage | 1.0 | 4.45s | 2 | Yes | Yes | — |
| 2 | M2 | Marriage | 1.0 | 4.32s | 3 | Yes | Yes | — |
| 3 | M3 | Marriage | 0.9 | 4.32s | 2 | Yes | No | — |
| 4 | M4 | Marriage | 1.0 | 4.32s | 2 | Yes | Yes | — |
| 5 | M5 | Marriage | 0.8 | 4.32s | 3 | Yes | No | — |
| 6 | M6 | Marriage | 0.8 | 4.31s | 3 | Yes | No | — |
| 7 | V1 | Venus | 0.8 | 4.33s | 3 | Yes | No | — |
| 8 | V2 | Venus | 0.7 | 4.32s | 4 | Yes | No | — |
| 9 | H1 | Horary | 0.9 | 4.31s | 2 | Yes | No | — |
| 10 | H2 | Horary | 0.8 | 4.31s | 2 | Yes | Yes | — |
| 11 | F1 | Financial | 0.9 | 4.32s | 3 | Yes | Yes | — |
| 12 | F2 | Financial | 0.8 | 4.32s | 3 | Yes | Yes | — |
| 13 | T1 | Timing | 1.0 | 4.31s | 2 | Yes | Yes | — |
| 14 | T2 | Timing | 0.9 | 4.31s | 2 | Yes | Yes | — |
| 15 | TR1 | Transit | 0.7 | 4.33s | 2 | Yes | Yes | — |
| 16 | TR2 | Transit | 0.7 | 4.32s | 2 | Yes | No | — |
| 17 | E1 | Edge | 0.8 | 4.31s | 1 | No | No | — |
| 18 | E2 | Edge | 0.6 | 4.31s | 2 | Yes | Yes | — |
| 19 | E3 | Edge | 0.7 | 4.31s | 1 | Yes | Yes | — |
| 20 | Q1 | Quality | 0.5 | 4.31s | 2 | Yes | Yes | — |
| 21 | Q2 | Quality | 0.5 | 4.33s | 2 | Yes | Yes | — |

## Detailed Results

### M1 — Marriage (weight: 1.0)

**Question (truncated):** KP chart: 7th cusp sub-lord is VEN (282-37-46, rashi SAT, nak MON, sub MAR).
Venus signifies houses: 1,4,6,9,12. House 7 significators: MER,SUN. House 2 sig: JUP,SUN. House 11 sig: JUP,MAR,MER,SAT,SUN...

**Expected format:** {"result":"Yes/No","explanation":"...","rule":"KP_MAR_xxxx","confidence":"high/medium/low"}

**Model answer:**

```
**Verdict:** NO

**Sub-lord Analysis:** The 7th cusp sub-lord Venus holds primary authority over marriage matters. However, its signification pattern creates a fundamental contradiction with marriage prospects.

**Rule Application:** Since Venus (the 7th sub-lord) signifies the 6th house (disputes, obstacles) along with the 12th house (losses, separation), it directly opposes the marriage-promoting combinations of houses 2, 7, and 11 as stated in the grounding rule.

**Confidence Level:** HIGH

The rule clearly states that if the 7th sub-lord connects with 6th or 12th house significations instead of benefic ones like 2nd, 7th, or 11th, marriage gets blocked. Here, Venus's dual connection to both dispute-causing 6th and loss-inducing 12th houses overrides any potential positive influences.

**Practical Implication:** This planetary configuration suggests inherent challenges in marriage materialization - whether through relationship disputes, family opposition, or circumstances leading to losses related to partnerships.

This demonstrates how precise sub-lord analysis combined with proper rule application provides definitive answers about life events rather than ambiguous predictions
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### M2 — Marriage (weight: 1.0)

**Question (truncated):** KP chart: 7th cusp sub-lord is VEN, signifying houses 1,4,6,9,12.
Hypothetical: VEN is retrograde. Dasa balance at birth: MAR 0Y 7M 23D.

According to KP: If the 7th sub-lord is retrograde, does it in...

**Expected format:** {"timing":"...","quote":"..."}

**Model answer:**

```
**Answer:** Marriage will be **denied completely**, not delayed or denied initially with possibility later.

**Quote & Rule ID:** "if the sub-lord of the 7th cusp is retrograde and it is deposited in the constellation of a retrograde planet, he can never marry at all" - This is the definitive statement from KP principles.

**Analysis:** Since Venus (VEN) as 7th cusp sub-lord signifies houses 1,4,6,9,12 without any benefic house connections (2,7,11), this already creates significant obstacles to marriage. The retrograde nature further strengthens the denial principle outlined in KP rules.

**Timing Conditions:** With no marriage promise due to retrograde sub-lord, other dasha periods become irrelevant for marriage timing. However, transits of Jupiter over natal Moon/Venus might create temporary relationship opportunities that ultimately fail to materialize into marriage.

The key distinction here is between **delay** versus **denial**: KP clearly states when retrograde 7th sub-lord lacks beneficial house significations, marriage becomes impossible rather than merely postponed during favorable planetary periods.

Confidence: high
rules_used: KP_MAR_0969, KP_MAR
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### M3 — Marriage (weight: 0.9)

**Question (truncated):** KP chart: 7th cusp sub-lord is VEN, signifying houses 1,4,6,9,12.
Hypothetical: Suppose VEN signified houses 1,7,9 simultaneously.

According to KP: What effect does a sub-lord being significator for ...

**Expected format:** explanation quoting book rule

**Model answer:**

```
**Answer:** Marriage is **denied** according to this configuration.

**Step-by-Step Analysis using KP Rules:**

The 7th cusp sub-lord VEN signifies houses 1,4,6,9,12. Let's examine each:

**House Significations:**
- Houses 2, 7, 11 = **Marriage indicators** (not present here)
- Houses 4, 6, 10 = **Denial factors** (present as 4, 6, 12)

**Rule Application:**
Since VEN doesn't signify houses 2, 7, 11 (marriage houses), we look at denial combinations. The presence of houses 4, 6, 12 creates a strong denial pattern.

**1st House Effect:**
As per KP_MAR_0643, the 1st house represents "self" who desires partnership. When its significator denies marriage, it weakens personal desire rather than strengthening it.

**9th House Impact:**
The grounding rule mentions that 9th house significators create complications. In traditional astrology, 9th house relates to dharma and higher wisdom. Here, it acts as a hindrance factor
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### M4 — Marriage (weight: 1.0)

**Question (truncated):** KP chart: House 7 significators: MER,SUN. Additional candidates: VEN(7th sub-lord), JUP(2nd sig), MAR(11th sig).
Planet significators — Sun:4,7,9,11,12. Mercury:5,7,8,9,11,12. Venus:1,4,6,9,12. Jupite...

**Expected format:** single planet chosen + step-by-step elimination

**Model answer:**

```
**Answer:** When multiple planets signify marriage, we must identify the **primary significator** using KP's hierarchical ranking system.

**Quote:** "Rank: Sub-lord of the portion occupied by the person denoted by the house, Occupant of the house, Lord of the house, and the Star-lord of the occupying planet." - KP_PRO_1045

**Rule ID:** KP_PRO_1045

**Confidence:** high

In your case, **Venus (7th sub-lord)** becomes the primary significator since it holds the highest rank according to this hierarchy:

1. **Sub-lord**: Venus (as 7th sub-lord)
2. **Occupant**: None (no direct occupants mentioned)
3. **Lord**: Not applicable here
4. **Star-lord**: Depends on which nakshatra Venus occupies

The sub-lord carries maximum importance because it determines whether the promise will actually manifest during its dasha/antardasha periods. While other planets like Sun and Mercury may be strong significators due to their conjunctions, Venus maintains primacy due to its role as 7th sub-lord.

For accurate prediction timing, prioritize Venus periods over others, though all qualifying
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### M5 — Marriage (weight: 0.8)

**Question (truncated):** KP chart: 7th cusp nak-lord is Mars. Mars signifies houses 1,3,10,11,12.
Hypothetical: Mars is retrograde AND debilitated (in Cancer).

According to KP: What is the effect of a retrograde+debilitated ...

**Expected format:** delay/deny/complication + rule quote

**Model answer:**

```
**Answer:** Complete denial of marriage according to KP principles.

**Quote:** "If one wants to know whether marriage is promised at all and if the sub-lord of the 7th cusp is retrograde and it is deposited in the constellation of a retrograde planet, he can never marry at all." - **KIP_MAR_0971**

This specific combination creates what KP terms as a "marriage-blocking" scenario. The retrograde nature already indicates obstacles and delays in matrimonial matters. When this retrograde planet is also debilitated, it becomes extremely weak and unable to deliver positive results for marriage.

**Debilitation Treatment in KP vs Vedic:**
Unlike traditional Vedic astrology where debilitation might simply weaken planetary strength, KP views debilitation as a fundamental obstruction. In KP, the constellation lord's condition significantly impacts the planet's ability to grant results. A debilitated planet loses its capacity to function normally, regardless of its house position or aspects.

**Outcome Analysis:**
The rule clearly states "he can never marry at all," indicating complete denial rather than mere delays. This is because both the nakshatra lord (Mars) and the depositor planet are afflicted simultaneously, creating insur
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### M6 — Marriage (weight: 0.8)

**Question (truncated):** KP chart: 7th cusp sub-lord is VEN, signifying houses 1,4,6,9,12.
VEN signifies 6th(disputes/separation) and 12th(losses). Native had 5 broken engagements in 8 years.

According to KP: What explains r...

**Expected format:** list of stated reasons from KP

**Model answer:**

```
**Answer:** Repeated failed engagements are explained by the strong connection between the 7th cusp sub-lord (Venus) and the 6th house, combined with its link to the 12th house.

**Quote & Analysis:**
The native's pattern of five broken engagements in eight years perfectly aligns with KP principles. Venus as the 7th cusp sub-lord being a significator of both 6th and 12th houses creates a powerful combination for relationship failures.

**House Significance:**
- **6th House**: Represents disputes, legal battles, separation, enemies - directly causing engagement breakdowns
- **12th House**: Indicates losses, endings, dissolution - confirming relationships don't materialize permanently

**Rule Application:**
This scenario directly matches KP_MAR_0643 where "a person is denied of marriage" when 7th cusp sub-lord connects with 6, 10 (separation/disputes), making permanent union impossible despite initial attraction.

**Remedy Timing:**
Favorable periods would occur during dashas/bhuktis of planets that are NOT significators of 6, 12 houses but rather support 2nd, 7th,
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### V1 — Venus (weight: 0.8)

**Question (truncated):** KP chart: Venus at 282-37-46 in Aquarius(SAT rashi), Dhanishta nak(MON star), sub MAR.
Venus signifies: 1,4,6,9,12. Moon signifies: 1,3,6,10. Mars signifies: 1,3,10,11,12.

As natural karaka for marri...

**Expected format:** verdict + book quote

**Model answer:**

```
Answer: Venus as sub-lord cannot deliver marriage despite being karaka due to conflicting significances.

**Quote:** "If the sub-lord happens to be Venus, one will be writing much about the marriage, sex, love affairs, Chemistry etc." - [KP_MAR_1016]

**Rule ID:** KP_MAR_1016

**Analysis:** Venus is positioned in Moon's nakshatra (Dhanishta). Since Moon signifies the 6th house along with other beneficial houses (1st, 3rd, 10th), there appears to be some connection. However, the critical factor here is Venus itself signifying houses 1, 4, 6, 9, 12, which includes the detrimental 6th house representing separation and disputes.

According to KP principles, when a planet acts as both karaka and sub-lord but carries conflicting energies through its significations, the negative influences prevail. The 6th house significance completely overrides any potential positive results Venus might offer.

**Mars Sub-lord Influence:** Mars signifying 11th house brings gains and fulfillment of desires. This creates an interesting dynamic where while Venus may deny marriage due to its 6th house connection,
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### V2 — Venus (weight: 0.7)

**Question (truncated):** According to KP astrology, what is the difference when Venus is:
(a) In the 5th bhava (love affairs, romance)
(b) In the 7th bhava (marriage, partnership)

What does Venus signify differently in each?...

**Expected format:** difference in interpretation + exact lines

**Model answer:**

```
**Venus in Different Bhavas - Significances**

When Venus occupies the **5th bhava**, it primarily signifies "romance, love affairs and children" as per KP principles. The native experiences strong romantic inclinations and creative expression through relationships.

However, when Venus moves to the **7th bhava**, its signification transforms entirely toward "marriage, partner and business partnerships." This fundamental shift occurs because the 7th house governs all forms of committed partnerships.

**Sub-Lord Modification Effect**

The crucial factor determining whether Venus manifests as mere love or actual marriage lies with its **sub-lord**. According to KP_MAR_1005, if Venus (as 7th cusp sub-lord) simultaneously becomes a significator of the 11th house, this creates a powerful combination for marriage fulfillment. The 11th house represents gains, achievements, and realization of desires.

In contrast, when Venus operates solely as a 5th house influence without 7th house connection, it remains limited to romantic expression rather than matrimonial commitment.

**Practical Application**
A native with Venus in 5th house may experience multiple passionate relationships, but marriage requires additional planetary support connecting the 7th
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### H1 — Horary (weight: 0.9)

**Question (truncated):** A querent selects horary number 147 for a marriage question.

According to KP horary method: How to determine lagna degree from number 147?
What are the ruling planets? Show step by step: ascendant si...

**Expected format:** list of ruling planets + method trace

**Model answer:**

```
**Step-by-Step Horary Calculation for Number 147:**

**1. Ascendant Degree Determination:**
For number 147, add 6° (the fixed cusp position mentioned). The resulting degree becomes the lagna position. This follows the specific calculation method outlined in KP principles where we add 6° to arrive at the actual ascending degree at moment of judgment.

**2. Ruling Planets Extraction:**

**Ascendant Sign Lord:** Determine which planet governs the sign containing the calculated lagna degree
**Star Lord:** Find the nakshatra ruler for that degree
**Sub-Lord:** Calculate using Krishnamurti's precise subdivision system

**Moon Position Calculations:**
Similarly calculate Moon's sign, nakshatra, and sub-lords using the same degrees.

**Day Lord:** Establish which planetary period (dasha/antardasha/pratyantar) was operating on that specific date.

**Marriage Confirmation/Denial Methodology:**

The ruling planets must connect with houses 2, 7, 11 through ownership, occupation, or stellar connection. House 2 represents family addition, house 7 indicates spouse/partnership, while house 11 confirms fulfillment of desires.
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### H2 — Horary (weight: 0.8)

**Question (truncated):** KP horary: 7th house significators found — Sun(occupant), Mercury(sub-lord connection), Venus(karaka+7th sub-lord), Saturn(depositor chain), Rahu(star of 7th occupant).
Some are benefic, some malefic ...

**Expected format:** tie-breaker rules + which planets to drop

**Model answer:**

```
**Answer:** Drop Saturn and Rahu as they cause delays and obstacles in marriage matters.

**Quote:** "If you find that the significators include both beneficial and detrimental influences, then you must eliminate the harmful ones... You will have to drop those which are detrimental."

**Rule ID:** KP_PRO_0690 (Significator elimination)

**Confidence:** high

In KP horary analysis, when multiple significators appear with mixed benefic/malefic nature, specific planetary combinations require exclusion based on their inherent qualities affecting marriage outcomes.

Saturn and Rahu typically create delays, restrictions, and karmic obstacles - even if they're strong significators by occupation or sub-lordship connections. Their presence can manifest as:

- Delayed marriage timing
- Family opposition
- Financial constraints
- Health issues affecting marriage prospects

The remaining significators (Sun, Mercury, Venus) should be evaluated through their strength, dignity, and connectivity to houses 2, 7, 11 without interference from dropped malefics like Saturn-Rahu combination.

This selective approach ensures accurate prediction while eliminating planetary influences that could undermine positive marriage indications.

**Reference:** Significator elimination principle in KP_PRO_0690

**Additional Note:**
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### F1 — Financial (weight: 0.9)

**Question (truncated):** KP chart: 11th cusp sub-lord is MAR. Mars signifies houses: 1,3,10,11,12.
House 2 sig: JUP,SUN. House 11 sig: JUP,MAR,MER,SAT,SUN.

According to KP: Will native experience financial gains? Mars signif...

**Expected format:** {"gains_likely":"Yes/No","timing":"...","rule":"..."}

**Model answer:**

```
**Answer**: Yes, native will experience financial gains based on KP principles.

**Quote**: "The sub lord of the cusp of 11 is connected with any one of the houses 2, 6, 10 and 11 in any manner, the consultant will have the positive answer."

**Rule ID**: KP_PRO_0479

**Confidence**: high

In your case, Mars as 11th cusp sub-lord connects to multiple beneficial houses - 1st (self), 3rd (efforts), 10th (profession), 11th (gains). This creates a strong foundation for financial success through personal initiatives and professional endeavors.

Regarding Mars signifying both 11th and 12th houses, KP treats this as dual influence rather than contradictory outcomes. The stronger signification typically dominates. Since Mars primarily signifies gain-giving houses (1,3,10,11), the negative 12th house effect gets overshadowed.

For timing, examine when Mars dasha/bhukti operates or when transiting planets activate Mars's nakshatra position. Additionally, analyze Jupiter-Sun periods since these signify the 2nd house (wealth accumulation).

The role of 2nd house connections
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### F2 — Financial (weight: 0.8)

**Question (truncated):** KP chart: 11th sub-lord MAR signifies 1,3,10,11,12 (includes 12th=losses).
8th sub-lord RAH signifies 3,10,12.

According to KP: When 11th sub-lord connects to 12th house, what happens to gains?
When ...

**Expected format:** explanation for blocked gains + exact quote

**Model answer:**

```
**Answer Analysis**
The 11th sub-lord MAR connecting to houses 1,3,10,11,12 presents a complex scenario where gains are simultaneously promoted and obstructed. Since MAR signifies both beneficial houses (1,3,10,11) and the 12th house of losses, this creates a contradictory influence on income potential.

Similarly, the 8th sub-lord RAH connecting to 3,10,12 indicates transformation through obstacles. The 8th house represents sudden changes and hidden factors affecting wealth accumulation.

**Rule Application**
Following KP_FIN_0585, when the 11th sub-lord MAR occupies signs 6/8/12 away from its owned houses while simultaneously signifying the 12th house, gains become "difficult of acquisition" - meaning blocked or severely restricted rather than merely delayed.

For the 8th sub-lord RAH connection with 12th house significance, KP principles suggest these combinations create situations where resources flow toward others' interests instead of personal gain, indicating redirection of expected benefits.

**Confidence:** high
**Rules Used:** KP_FIN_0585, KP_MAR_1112, KP_MAR_1126
**Redirected Benefits:**
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### T1 — Timing (weight: 1.0)

**Question (truncated):** KP chart: Marriage houses are 2,7,11. 7th cusp sub-lord is VEN(signifies 1,4,6,9,12).
House 7 sig: MER,SUN. House 2 sig: JUP,SUN. House 11 sig: JUP,MAR,MER,SAT,SUN.

According to KP: Must BOTH Mahadas...

**Expected format:** quote + Yes/No

**Model answer:**

```
**Answer:** Both Mahadasha and Antardasha lords must be connected to houses 2, 7, and 11 for marriage manifestation.

**Quote & Rule ID:** "Marriage will take place during the conjoined periods of the significators for houses 2, 7 and 11..." - [KP_TIM_0660]

In KP system, both levels of dashas operate simultaneously. The Mahadasha lord acts as the primary timing agent while the Antardasha lord provides the specific window. For successful marriage materialization, both planets should ideally signify the same marriage houses (2-7-11). However, sometimes one may be stronger significator than other.

**Chart Analysis:** 
Mahadasha candidates: Jupiter (signs 2, 11), Sun (signs 2, 7, 11)
Antardasha candidates: Multiple options exist based on current period

The most favorable combination would involve **Jupiter-Sun**, **Sun-Jupiter**, or similar strong significator pairs where both planets connect to all three marriage houses through direct ownership or occupation.

**Confidence:** high
**Rules Used:** KP_TIM_0660

Note: Exact planetary combinations depend on running dasha sequence at
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### T2 — Timing (weight: 0.9)

**Question (truncated):** KP chart: Native runs Jupiter Mahadasha. Jupiter signifies: 2,5,11,12 (connected to 2,11 — favorable).
Current Antardasha: Ketu. Ketu signifies: 5,6,8,12 (NOT connected to 2,7,11).

According to KP: W...

**Expected format:** expected outcome/timing

**Model answer:**

```
**Answer:** Marriage will be **delayed** during this current Ketu Antardasha period.

**Quote:** "When we follow the principles that Jupiter in 2, 5, 7, 9 and 11th houses alone can give fruitful results, how can marriage take place when Jupiter is in the third house?"

**Rule ID:** KP_TIM_0536

**Confidence:** high

In your case, Jupiter (Maha lord) signifies beneficial houses for marriage (2nd, 11th), while Ketu (Antar lord) signifies non-marriage houses (6th, 8th, 12th). This creates a conflicting energy where the promise exists at the major level but gets blocked at the sub-level.

The grounding rule clearly states that even if Jupiter shows marriage potential due to its connection with houses 2, 7, 11, it cannot deliver until the sub-period ruled by Saturn within Jupiter's dasha concludes - because Saturn doesn't support marriage through similar negative significances.

For optimal timing, wait for an Antardasha ruler that simultaneously supports marriage (signifies 2, 7, 11) while avoiding opposition from houses like 6, 8, 12 which
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### TR1 — Transit (weight: 0.7)

**Question (truncated):** KP chart: 7th sub-lord Venus is at 282-37-46 (Aquarius).
Saturn (malefic) transits over Venus's natal position during a favorable dasha for marriage.

According to KP: Does malefic transit over sub-lo...

**Expected format:** stated effect on timing/outcome

**Model answer:**

```
**Answer:** Yes, Saturn's transit will create delays despite the favorable dasha period.

**Quote:** "Though the matter relates to marriage...the various events about marriage is clearly shown by the sub, transited by the significator."

**Rule ID:** KP_MAR_0599

In this scenario, we have the 7th cusp sub-lord Venus positioned at Aquarius degree 282-37-46. When Saturn (being a natural malefic) transits over this precise natal position of Venus, it creates obstacles that manifest as timing delays rather than complete denial of marriage.

The key principle here is that while the dasha provides the broad time frame when an event can occur, the actual timing depends on finer planetary influences - specifically which planet acts as the significator currently transiting through sensitive points like the sub-lord's birth position.

Saturn's slow-moving nature ensures its influence persists throughout its transit duration. However, since you mentioned a "favorable dasha" already supporting marriage prospects, this indicates the basic promise exists. The transit merely fine-tunes the exact timing within that broader favorable period.

This demonstrates how KP integrates both macro-level dashas with micro-level transits for precise predictions.

Confidence
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### TR2 — Transit (weight: 0.7)

**Question (truncated):** KP chart: 7th cusp at 122-12-49 (Leo). 7th sub-lord VEN is weak (signifies 6th,12th).
Jupiter (benefic) transits over 7th cusp degree. Dasha is marginally favorable.

According to KP: Can benefic tran...

**Expected format:** mitigation rules

**Model answer:**

```
Answer: **No**, benefic transit cannot rescue unfavorable marriage timing when sub-lord is weak according to KP principles.

**Quote**: "If the significator of 6th house is Jupiter and it has entered in Poorvapalguni (133-20'to 146-40') constellation, till it touches Venus Sub and Mercury Sub-sub or Mercury Sub in Venus Star, one cannot be promoted..."

**Rule ID**: [KP_HEA_0837]

This fundamental KP principle demonstrates how transit influence operates secondary to natal promise indicated by sub-lords. The key phrase is "till it touches" - meaning the beneficial effect remains blocked until specific stellar combinations align.

In your case with 7th sub-lord Venus being weak (signifying 6th and 12th houses), any positive transit would face similar restrictions. The benefic nature of Jupiter's transit becomes irrelevant when it must wait for supportive nakshatra divisions within its current star position.

**Transit vs Sub-lord Primacy**: In KP system, sub-lord holds supreme authority determining whether an event will manifest favorably. Benefic transits act as supporting agents, but cannot override negative sub-lord promises unless they reach specific stellar positions where
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### E1 — Edge (weight: 0.8)

**Question (truncated):** KP chart marriage analysis (houses 2,7,11):
Sun: sig 4,7,9,11,12 (7+11 favorable, 12 unfavorable)
Mercury: sig 5,7,8,9,11,12 (7+11 favorable, 8+12 unfavorable)
Venus(7th sub-lord): sig 1,4,6,9,12 (NOT...

**Expected format:** step-by-step evaluation + final verdict

**Model answer:**

```
**Step-by-Step Evaluation:**

The given planetary positions reveal significant challenges for marriage timing:

1. **Sun Analysis**: While Sun signifies beneficial houses 7 and 11, its connection to house 12 creates obstacles through separation tendencies.

2. **Mercury Assessment**: Mercury's dual role as both benefic (signifying 7, 11) and malefic (connected to 8, 12) creates internal contradictions in its influence.

3. **Critical Venus Examination**: Most importantly, Venus as the 7th sub-lord fails to connect with either house 7 OR house 11 - this is the decisive factor according to KP principles.

**Priority Determination:**
In KP methodology, the sub-lord holds supreme authority over results. Since Venus governs the 7th cusp but lacks essential connections to marriage houses (7, 11), it becomes non-cooperative regarding matrimonial prospects.

**Marriage Verdict:**
According to KP principle that "if the sublord of the seventh is not connected with houses 2, 7 and 11 then he will deny marriage," the native faces severe impediments to marriage materialization during Venus periods/sub-periods.

This configuration suggests delayed or denied marriage unless
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### E2 — Edge (weight: 0.6)

**Question (truncated):** According to KP: Are there special rules when:
1. 7th cusp sub-lord is in a watery sign (Cancer/Scorpio/Pisces)?
2. 7th cusp itself falls in a watery sign?
3. Does sign element (fire/earth/air/water) ...

**Expected format:** special rules for water signs

**Model answer:**

```
**Answer:** Yes, specific rules apply regarding water signs and their impact on marriage.

**Quote & Rule IDs:**
"Kona houses are never evil to anyone born in them." - KP_PRO_0935
"The twelve houses to Kona houses indicate denial of their results." - KP_PRO_0935
"When the lords of 1st and 8th houses conjoinedly influence the matters of houses 3 and 12, danger to life is predicted." - KP_LON_0054

**Confidence:** high
**Rule Classification:** KP_PRO_0935 (Sign Element Effects)

**Analysis:**
Regarding your questions:

1. **Watery Sign Sub-Lord**: When the 7th cusp sub-lord occupies Cancer, Scorpio, or Pisces, it creates specific marital dynamics affecting emotional connections and intuitive partnerships.

2. **Direct Watery Fall**: If the 7th cusp itself falls in a watery sign, it influences the native's approach toward partnership and relationship handling.

3. **Elemental Influence**: Fire elements bring passion and quick decisions, earth provides stability, air introduces mental compatibility, while water emphasizes emotions and nurturing qualities in marriage assessment.

4. **Leo Position**:
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### E3 — Edge (weight: 0.7)

**Question (truncated):** According to KP: A planet is debilitated (e.g. Venus in Virgo) BUT in nakshatra of a benefic (Jupiter's star).
1. Does KP treat debilitation same as Vedic astrology?
2. Does nakshatra lord override de...

**Expected format:** mitigation or denial + quote

**Model answer:**

```
**Answer with Quotes & Rules:**

**1. Debilitation Treatment:** KP doesn't treat debilitation uniformly like traditional systems do.

"Debilitation means loss of strength and hence poor results during its period." - [KP_PRO_0105]

However, this quote alone cannot answer your question about uniform treatment.

**Confidence:** med
**Rule ID:** KP_PRO_0105

**2. Nakshatra Lord Override:** Yes, in KP, the nakshatra lord significantly influences planetary results.

"The constellation lord is more important than the house lord in determining the result." - [KP_PRO_0598]

This principle suggests the nakshatra lord can modify debilitation effects.

**Confidence:** high
**Rule ID:** KP_PRO_0598

**3. Sub-Lord Analysis:** For marriage timing through 7th cusp sub-lord:

"If the sub-lord is deposited in the constellation of significators of houses 6,10 then marriage gets denied." - [KP_MAR_0696]

If debilitated planet is 7th sub-lord in Jupiter's nakshatra, we need to check Jupiter's role as marriage significator.

**Confidence:** low
**Rule
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### Q1 — Quality (weight: 0.5)

**Question (truncated):** For rule citation fidelity: What is the exact content of KP rules KP_MAR_0673 and KP_MAR_0971?
What chapter/section of KP Reader do they come from? Are they from KP Reader I,II,III,IV,V, or VI?
Provid...

**Expected format:** verbatim quote with page/section

**Model answer:**

```
Based on the provided grounding rule, I must clarify that I don't have direct access to specific rule numbers like KP_MAR_0673 or KP_MAR_0971 without referencing the complete KP literature collection.

However, following the prescribed format strictly:

**Confidence:** low
**Answer:** Not available for these specific rule IDs
**Rule ID Reference:** None (cannot verify)
**Chapter/Section Location:** Unknown (requires full reference)

The original instruction mandates using "ONLY excerpts below" while citing rule IDs from provided texts. Since we're missing those references, I can only report that accurate rule citations require accessing the complete KP library or providing the actual rule texts for verification purposes.

If you could provide the actual rule contents or specify their locations in the KP Reader series, I'd be happy to assist further.

**Confidence Level:** low due to incomplete data availability

**Additional Request:** Please provide the complete rule texts or clarify their chapter locations if possible.

I'm constrained by the strict formatting guidelines requiring precise rule ID correlations before delivering any meaningful responses.

**Rule ID Verification Status:** INCOMPLETE DATA

Please refer to the grounding rule's emphasis on "NEVER invent pages/chapters" and "use ONLY excerpts below."

**Conf
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

### Q2 — Quality (weight: 0.5)

**Question (truncated):** When you cite rules like "KP_MAR_0673", is this an actual rule ID from KP Readers by Prof. K.S. Krishnamurti?
Or internally generated from training data? How confident are you in citation accuracy?
Ca...

**Expected format:** Yes/No + evidence

**Model answer:**

```
I must clarify that when I provide rule citations like "KP_MAR_0673" or "KP_EDU_0065," these are **not actual rule IDs** from any official KP publication. Instead, these appear to be **internally generated codes** created during my training process on KP texts.

My primary reference sources include:
- "Krishnamurti Paddhati Reader Volume I"
- "Krishnamurti Paddhati Reader Volume IV" (Marriage-specific)
- Various other KP publications and articles

However, upon closer examination, I realize that I generate unique identifiers ("rule IDs") rather than using established KP system numbers. This creates potential confusion about authenticity.

Regarding **citation accuracy**, my confidence level is **medium-low** because I'm generating these identifiers programmatically based on keyword patterns found in KP literature. While I strive to capture essential concepts accurately, there may be minor variations between what I extract versus the original texts.

For distinguishing different volumes - such as **Reader I (general)** vs **Reader IV (marriage)** - I rely on contextual keywords within each excerpt. However, without direct access to the physical publication structure, I cannot guarantee precise volume identification.

**Confidence in matching
```

**Book answer:** _TO BE FILLED_

**Rule ref:** _TO BE FILLED_

**Score:** TO BE SCORED

**Notes:** 

---

## Scoring Rubric

- **1.0 Exact-match:** model answer matches book passage word-for-word
- **0.75 Strong-match:** model paraphrases but keeps all logical conditions, correct rule ID
- **0.5 Partial:** partially correct but misses important clause/caveat
- **0.0 Mismatch:** model contradicts book or fabricates rule

### Penalties
- Missing rule citation: -0.1
- Incorrect application: -0.15
- Hallucinated rule: -0.25
