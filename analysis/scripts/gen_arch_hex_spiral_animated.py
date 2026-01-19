import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters
R = 1.0 / 2
d = 0.2
N = math.ceil((2 * math.pi * R**2) / (math.sqrt(3) * d**2))
N_revolutions = math.ceil(N / 6)

# Define axial directions for hex grid
directions = [
    (1, 0), (0, 1), (-1, 1),
    (-1, 0), (0, -1), (1, -1)
]

# Function to convert axial to Cartesian coordinates (rotated 90 degrees)
def axial_to_cartesian(q, r, size):
    x = size * math.sqrt(3) * (r + q / 2)
    y = size * (3/2 * q)
    return x, y

# Generate hexagonal spiral coordinates
def generate_hex_spiral(n_revolutions):
    coords = [(0, 0)]
    q, r = 0, 0
    for k in range(1, n_revolutions + 1):
        q += directions[4][0]
        r += directions[4][1]
        for i in range(6):
            for _ in range(k):
                coords.append((q, r))
                q += directions[i][0]
                r += directions[i][1]
    return coords

# Generate the spiral coordinates
spiral_coords = generate_hex_spiral(N_revolutions)

# Convert axial coordinates to Cartesian coordinates
cartesian_coords = [axial_to_cartesian(q, r, d) for q, r in spiral_coords]
x_coords = [x for x, y in cartesian_coords]
y_coords = [y for x, y in cartesian_coords]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
margin = d * 3
ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
ax.set_title('Hexagonal Spiral Scan Pattern')
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.grid(True)

# Initialize line and point objects
line, = ax.plot([], [], 'o-', color='blue', markersize=4)
texts = []
beam_segments = []

# Initialization function
def init():
    line.set_data([], [])
    for segment in beam_segments:
        segment.remove()
    beam_segments.clear()
    return line,

# Animation function
def animate(i):
    line.set_data(x_coords[:i+1], y_coords[:i+1])
    # Add text label above the current point
    if i < len(cartesian_coords):
        x, y = cartesian_coords[i]
        text = ax.text(x, y + 0.05, str(i+1), fontsize=8, ha='center', va='bottom')
        texts.append(text)
    # Draw beam segment
    if i > 0:
        x0, y0 = cartesian_coords[i-1]
        x1, y1 = cartesian_coords[i]
        beam_segment, = ax.plot([x0, x1], [y0, y1], '-', color='blue', linewidth=30, alpha=0.25, solid_capstyle='round', zorder=1)
        beam_segments.append(beam_segment)
    return line, *beam_segments, *texts

# Create the animation
ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=len(x_coords), interval=200, blit=True, repeat=False
)

# Save the animation as a GIF
output_dir = "analysis"
os.makedirs(output_dir, exist_ok=True)
ani.save(os.path.join(output_dir, "arch_hex_spiral_animation.gif"), writer='pillow', dpi=100)

# Save the final frame as PNG and PDF
fig.savefig(os.path.join(output_dir, "arch_hex_spiral.png"), format="png", bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(output_dir, "arch_hex_spiral.pdf"), format="pdf", bbox_inches='tight')

plt.show()
