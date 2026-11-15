"""Pick-and-place feasibility in the (G, K, D, A) model.

    problem.py    the problem: six rules, the text of K, the answer A
    generate.py   the dataset (python -m adapt.generate), and the larger training file of H6 (extra_train)
    verify.py     the single validity check of a row
    prompts.py    the prompt levels (full to compressed), the control levels, the progressive schedules
    kaggle.py     two-GPU lanes used to run the experiments on Kaggle
"""
