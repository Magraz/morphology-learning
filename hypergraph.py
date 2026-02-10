import matplotlib.pyplot as plt
import dhg

# draw a graph
g = dhg.random.graph_Gnm(10, 12)
g.draw()
# draw a hypergraph
hg = dhg.random.hypergraph_Gnm(10, 8)
hg.draw()

# Save the plot to a file
plt.savefig("my_plot.png", dpi=300, bbox_inches="tight")  # Saves a high-res PNG file

# Display the plot (call after savefig to avoid saving a blank image)
plt.show()
