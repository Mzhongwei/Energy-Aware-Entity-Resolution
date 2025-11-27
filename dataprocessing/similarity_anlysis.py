import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

def similarity_analysis(sim_list, actual_matches, output_file_name, k_first):
    x_vals_red = []
    y_vals_red = []
    x_vals_blue = []
    y_vals_blue = []
    for item in sim_list:
        # value ex. [[0.8194172382354736, 'idx__38260'], [0.8130440711975098, 'idx__51652']]
        values = sim_list[item]
        for value in values:
            record_num = int(value[1].split('__')[1])
            sim_degree = value[0]
            if record_num > item:
                el = (item, record_num)
            else:
                el = (record_num, item)
            if el in actual_matches:
                x_vals_red.append(el[0])
                y_vals_red.append(sim_degree)
                # point: (x:item, y:sim_degree) color: red 
            else:
                x_vals_blue.append(el[0])
                y_vals_blue.append(sim_degree) 
                # point: (x:item, y:sim_degree) color: blue
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals_blue, y_vals_blue, color='blue', label='Not Match')
    plt.scatter(x_vals_red, y_vals_red, color='red', label='Match')

    plt.xlabel('Item')
    plt.ylabel('Similarity Degree')
    plt.title('Similarity Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"pipeline/stat/sim_degree-{output_file_name}-{k_first}.png", dpi=300)

    # plt.show()
    plt.close()