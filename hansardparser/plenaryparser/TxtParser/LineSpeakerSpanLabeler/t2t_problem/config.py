# remove Flatworld tags from text.
RM_FLATWORLD_TAGS = False
# upsample rare classes.
UPSAMPLE = True
# resample data by this factor.
UPSAMPLE_FACTOR = 10
# number of context lines before and after line to include in example
CONTEXT_N_LINES = 4
# number of times to call `split_lines` on the training data. If 0, `split_lines`
# is not called.
N_SPLIT_LINES_PASSES = 1
