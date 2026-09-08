# H7: dense annotation against the upstream event log (6000 episodes, 120 tasks, 5 policies)

tolerance 1.0 s, grab flicker burst 2.0 s. Log grab events 72,200 -> 33,834 de-flickered grab runs (2.1 per run). Stale fork-vocabulary flags on these files: 0 (vacuous, not used).

| kind | count | episodes | share of episodes |
|---|---|---|---|
| C1 machine place fail, log credits it | 61 | 60 | 1.0% |
| C2 machine place pass, log scores a failure | 1,564 | 925 | 15.4% |
| C3 machine pick pass, log never grabbed it | 3,547 | 1,751 | 29.2% |
| C4 log grab run, machine saw no contact | 7,681 | 2,644 | 44.1% |
| A1 success, machine place unknown: recording ends before rest | 831 | 831 | 13.9% |
| A2 log grab run, machine saw contact or a failed pick only | 12,976 | 3,505 | 58.4% |
| any contradiction (C1-C4) | | 3,652 | 60.9% |

## by policy

| policy | episodes | any contradiction | C1 | C2 | C3 | C4 | A1 | A2 | successes | successes ending unsettled |
|---|---|---|---|---|---|---|---|---|---|---|
| cosmos3 | 1200 | 61.8% | 24 | 459 | 764 | 1531 | 289 | 3161 | 421 | 289 |
| g05_droid | 1200 | 58.0% | 12 | 207 | 579 | 1486 | 85 | 2505 | 126 | 85 |
| gr00t | 1200 | 64.2% | 7 | 270 | 814 | 1090 | 95 | 2253 | 122 | 95 |
| molmoact2_droid | 1200 | 58.5% | 6 | 195 | 501 | 2216 | 119 | 2618 | 166 | 119 |
| pi05 | 1200 | 61.8% | 12 | 433 | 889 | 1358 | 243 | 2439 | 330 | 243 |

## top 10 tasks by contradictions per episode

| task | episodes | contradictions / ep | episodes with one | C1 | C2 | C3 | C4 | A1 | A2 | successes | in ledger |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FruitsOnPlateTask | 50 | 13.18 | 100% | 0 | 79 | 93 | 487 | 0 | 314 | 0 |  |
| ToolsPickingAllHammersTask | 50 | 11.58 | 96% | 0 | 27 | 8 | 544 | 0 | 273 | 0 |  |
| FruitsOnPlate3Task | 50 | 8.98 | 100% | 1 | 18 | 47 | 383 | 10 | 187 | 12 |  |
| CleanUpToysTask | 50 | 8.86 | 88% | 0 | 2 | 95 | 346 | 0 | 267 | 0 |  |
| ToolOrganizationTask | 50 | 7.38 | 96% | 1 | 22 | 81 | 265 | 1 | 364 | 1 |  |
| ClearOrganicObjectsTask | 50 | 7.28 | 98% | 0 | 1 | 24 | 339 | 0 | 234 | 0 |  |
| HammersInLeftBinTask | 50 | 6.98 | 92% | 0 | 26 | 104 | 219 | 0 | 323 | 1 |  |
| ElectronicsInBinTask | 50 | 6.40 | 100% | 2 | 25 | 70 | 223 | 0 | 74 | 0 |  |
| CubesAndBlocksInBinTask | 50 | 5.80 | 96% | 0 | 44 | 117 | 129 | 8 | 350 | 16 |  |
| ToolOrganizationBothTask | 50 | 5.72 | 96% | 0 | 34 | 102 | 150 | 0 | 359 | 0 |  |

ledger tasks in the top 10: 0 of 8 (none). H7 asks for >= 6 of 10.

## the ledger tasks, wherever they rank

| task | rank of 120 | contradictions / ep | C1 | C2 | C3 | C4 | A1 | A2 | successes |
|---|---|---|---|---|---|---|---|---|---|
| GreenSpoonsInPotTask | 11 | 5.46 | 0 | 54 | 90 | 129 | 2 | 104 | 2 |
| PickDrillTask | 59 | 1.34 | 0 | 0 | 18 | 49 | 0 | 87 | 0 |
| GrabAFruitTask | 72 | 1.08 | 0 | 0 | 21 | 33 | 0 | 7 | 2 |
| BlockStackingSpecifiedOrderTask | 100 | 0.42 | 0 | 0 | 16 | 5 | 0 | 77 | 0 |
| Stack3RubiksCubeTask | 107 | 0.36 | 0 | 0 | 14 | 4 | 0 | 46 | 4 |
| BlockStackingOrderAgnosticTask | 114 | 0.22 | 0 | 0 | 9 | 2 | 0 | 106 | 7 |
| BowlStackingLeftOnRightTask | 117 | 0.16 | 0 | 5 | 2 | 1 | 8 | 39 | 9 |
| BowlStackingRightOnLeftTask | 119 | 0.12 | 0 | 5 | 0 | 1 | 14 | 22 | 16 |

## C1 by task: scored successes the machine judged a failed place

| task | C1 | successes |
|---|---|---|
| BananasInBinOneMoreTask | 4 | 48 |
| BananasInBinThreeTotalTask | 4 | 49 |
| BananasInCrateTask | 4 | 34 |
| MouseOnKeyboardTask | 4 | 16 |
| PutTwoMugsOnShelfTask | 4 | 3 |
| SauceBottlesCrateTask | 4 | 40 |
| ToyInBinTask | 4 | 9 |
| RecycleCartonsVerticalCrateTask | 3 | 1 |
| ElectronicsInBinTask | 2 | 0 |
| FoodPacking3CansTask | 2 | 0 |
| FruitsOnionToPlateTask | 2 | 16 |
| AppleAndYogurtInBowlTask | 1 | 2 |
| BBQSauceInBinTask | 1 | 6 |
| BananaInBowlTask | 1 | 42 |
| BlackItemsInBinTask | 1 | 0 |
| BowlInBinTask | 1 | 33 |
| ButterAboveRaisinTask | 1 | 18 |
| ClampInRightBinTask | 1 | 1 |
| CoffeePotInBinTask | 1 | 2 |
| FoodPacking1BoxesTask | 1 | 4 |

## A1 by task: successes whose recording ends before the object is at rest

| task | A1 | successes |
|---|---|---|
| BananaOnPlateTask | 45 | 50 |
| RubiksCubeTask | 39 | 39 |
| BananasInBinOneMoreTask | 36 | 48 |
| RubiksCubeOrBananaTask | 36 | 36 |
| BananaInBowlTask | 35 | 42 |
| BananasInBinThreeTotalTask | 34 | 49 |
| SauceBottlesCrateTask | 33 | 40 |
| OneBottleInSquarePailTask | 32 | 32 |
| BowlInBinTask | 26 | 33 |
| FruitsOnionTask | 24 | 24 |
| UnstackRubiksCubeTask | 22 | 38 |
| BananasInCrateTask | 21 | 34 |
| FruitsMovingOrangeOrLimeTask | 20 | 20 |
| MustardInLeftBinTask | 20 | 26 |
| BananasOutOfBinTask | 19 | 20 |

