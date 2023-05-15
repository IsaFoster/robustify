from tqdm import tqdm
from ._filter import get_levels

def initialize_progress_bar(corruption_list, corruptions, df):
    total = 1 
    for item in list(corruption_list):
        feature_names, levels = get_levels(item, df)
        total += ((len(feature_names) * len(levels)) * corruptions)
    return tqdm(total=total, desc="Total progress: ", position=0)