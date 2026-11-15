# Hand check of 30 test rows

30 fixed rows of `test.jsonl` were printed with G, K and D only (no answer) and checked twice, independently of the code.

1. By hand: one author worked the six rules out on paper and wrote down "feasible" or "not feasible" for every row, then
   compared with the answers. **30/30 match** (feasible or not; the violated rules were not written down on paper).
2. By a large language model: the model read only the sheet and the six rules and worked every rule out in writing,
   without running code; its sheet was saved before it was compared with the answers. **30/30 match**. The same kind of
   model helped write the generator, so this pass does not replace the check by hand.
