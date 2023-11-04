import itertools

# define the possible variables
possible_learning_rates = [0.01, 0.001, 0.0001, 0.00001]
possible_batch_sizes = [8, 16, 32, 64]
possible_filter_types = ["Mel", "Bark"]
possible_filter_shapes = ["Square", "Triangle"]
possible_overlaps = [1 / 2, 1 / 3, 1 / 4]
possible_frame_sizes = [512, 1024, 2048]
possible_energy_included = [1,0]
possible_temporal_included = [1,0]
possible_no_bins = [32,64,128]
possible_truncate_no_bin = [8,16,24]

# generate a matrix of all possible combinations
combinations = list(
    itertools.product(
        possible_learning_rates,
        possible_batch_sizes,
        possible_filter_types,
        possible_filter_shapes,
        possible_overlaps,
        possible_frame_sizes,
        possible_energy_included,
        possible_temporal_included,
        possible_no_bins,
        possible_truncate_no_bin
    )
)

# Iterate through the combinations
for combination in combinations:
    learning_rate = combination[0]
    batch_size = combination[1]
    filter_type = combination[2]
    filter_shape = combination[3]
    overlap = combination[4]
    frame_sizes = combination[5]
    energy_included = combination[6]
    temporal_included = combination[7]
    no_bins = combination[8]
    truncate_no_bins = combination[9]
    

    print(learning_rate, batch_size, filter_type, filter_shape, overlap, frame_sizes, energy_included, temporal_included, no_bins, truncate_no_bins)
