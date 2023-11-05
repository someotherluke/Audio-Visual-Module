import itertools

#Define the possible variables
possible_learning_rates = [0.01, 0.001, 0.0001, 0.00001]
possible_batch_sizes = [8, 16, 32, 64]


#Generate a matrix of all possible combinations
combinations = list(
    itertools.product(
        possible_learning_rates,
        possible_batch_sizes,
    )
)

#Iterate through the combinations
def combination():
    for combination in combinations:
        learning_rate = combination[0]
        batch_size = combination[1]
        
        print(learning_rate, batch_size)

if __name__ == '__main__':
    combination()