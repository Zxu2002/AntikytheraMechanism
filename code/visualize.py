import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_path = "data/1-Fragment_C_Hole_Measurements.csv"  
    data = pd.read_csv(file_path)

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    unique_sections = data["Section ID"].unique()
    colors = plt.cm.get_cmap("tab10", len(unique_sections)) 


    for i, section in enumerate(unique_sections):
        section_data = data[data["Section ID"] == section]
        plt.scatter(section_data["Mean(X)"], section_data["Mean(Y)"], 
                    color=colors(i), label=f"Section {section}", s=50, edgecolors="k")


    # Label axes and title
    plt.xlabel("Mean X (mm)")
    plt.ylabel("Mean Y (mm)")
    plt.title("Measured Hole Locations in the X-Y Plane")
    plt.axis("equal")
    plt.legend(title="Section ID")
    plt.savefig("graphs/measured_hole_positions.png")
    plt.show()
