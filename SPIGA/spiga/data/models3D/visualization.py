import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Starting main function")
    # Input arguments control
    pars = argparse.ArgumentParser(description='3D model visualization')
    pars.add_argument('file', type=str, help='File txt path')
    args = pars.parse_args()
    visualize_3Dmodel(args.file)

def visualize_3Dmodel(input_file):
    print(f"Reading file: {input_file}")
    with open(input_file) as f:
        lines = f.readlines()

    model = []
    for line in lines:
        line = line.strip()  # Remove \n and any surrounding whitespace
        line_split = line.split('|')
        values = np.array(line_split, dtype=float)
        model.append(values)

    model = np.array(model)
    model_ids = model[:, 0]  # Assuming the first column is the ID
    model_xyz = model[:, 1:]
    print(f"Model data shape: {model_xyz.shape}")

    # Show model
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model_xyz[:, 0], model_xyz[:, 1], model_xyz[:, 2] + 0.8)

    # Add labels for each point
    for i in range(len(model_ids)):
        ax.text(model_xyz[i, 0], model_xyz[i, 1], model_xyz[i, 2] + 0.8, str(int(model_ids[i])))

    # Save the figure
    plt.savefig('3d_model_plot.png')
    print("Figure saved as 3d_model_plot.png")

    # Show the plot
    plt.show()
    print("Plot displayed")

if __name__ == '__main__':
    print("Script is being run directly")
    main()
else:
    print("Script is being imported")
