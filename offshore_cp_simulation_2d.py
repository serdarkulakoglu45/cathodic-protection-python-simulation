import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from scipy.sparse import linalg, csr_matrix

@dataclass
class OffshoreMaterial:
    name: str
    conductivity: float
    potential: float
    current_density: float
    thickness: float = 0.01

class OffshoreICCPSimulator2D:
    def __init__(self, grid_size=(200, 200), domain_size=(50, 20)):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.potential = np.zeros(grid_size)
        self.current_density = np.zeros(grid_size)
        self.anodes = []
        self.structure_points = []
        self.protection_threshold = -0.85
        self.coating_integrity = 0.85  # Changed from 0.70 to 0.85
        self.salinity = 50.0  # Changed from 35.0 to 50.0
        self.water_agitation = 0.75
        self.setup_materials()
        self.calculate_current_requirements()

    def setup_materials(self):
        # Calculate seawater conductivity based on salinity (UNESCO 1983 equation simplified)
        seawater_conductivity = 0.18 * self.salinity  # Approximate conductivity in S/m at 20°C
        
        self.materials = {
            'steel': OffshoreMaterial('Steel', 5.8e6, -0.85, 0.01, 0.05),
            'titanium': OffshoreMaterial('MMO/Ti', 2.4e6, -0.73, 100, 0.01),
            'seawater': OffshoreMaterial('Seawater', seawater_conductivity, 0.0, 0.0),
            'coating': OffshoreMaterial('Epoxy', 1e-10, 0.0, 0.0, 0.0005)
        }

    def add_structure(self):
        # Create 2D pipeline structure
        pipeline_y = 5.0  # Height from seabed
        pipeline_radius = 0.5  # Pipeline radius in meters
        
        # Main pipeline
        start_pos = np.array([5.0, pipeline_y])
        end_pos = np.array([self.domain_size[0] - 5.0, pipeline_y])
        self.structure_points.append({
            'start': start_pos,
            'end': end_pos,
            'thickness': pipeline_radius,
            'material': self.materials['steel']
        })
            
    def add_iccp_anodes(self, num_anodes=4):  # Changed from 3 to 4
        # Add ICCP anodes along the pipeline
        pipeline_y = 5.0
        anode_spacing = (self.domain_size[0] - 12.0) / (num_anodes - 1)
        
        for i in range(num_anodes):
            x_pos = 6.0 + i * anode_spacing
            self.anodes.append({
                'position': np.array([x_pos, pipeline_y + 0.8]),  # Anodes slightly above pipeline
                'current': 30.0,
                'material': self.materials['titanium']
            })

    def calculate_current_requirements(self):
        # Pipeline dimensions
        pipeline_length = self.domain_size[0] - 10.0
        pipeline_diameter = 1.0
        
        # Calculate bare area
        total_surface_area = np.pi * pipeline_diameter * pipeline_length
        bare_area = total_surface_area * (1 - self.coating_integrity)
        
        # Current density requirements based on salinity
        if self.salinity > 30.0:
            base_current_density = 10.0
        elif self.salinity > 20.0:
            base_current_density = 15.0
        else:
            base_current_density = 20.0
            
        # Adjust current density based on water agitation
        agitation_factor = 1 + self.water_agitation  # More agitation needs more current
        base_current_density *= agitation_factor
            
        salinity_factor = np.sqrt(35.0 / self.salinity)
        self.total_current_required = (bare_area * base_current_density * salinity_factor) / 1000.0
        
        # Print calculation details
        print("\nCathodic Protection Calculation Results:")
        print(f"Pipeline Length: {pipeline_length:.1f} m")
        print(f"Total Surface Area: {total_surface_area:.1f} m²")
        print(f"Bare Area (with {(1-self.coating_integrity)*100:.1f}% coating damage): {bare_area:.1f} m²")
        print(f"Water Agitation Factor: {agitation_factor:.2f}")
        print(f"Base Current Density: {base_current_density:.1f} mA/m²")
        print(f"Salinity Factor: {salinity_factor:.2f}")
        print(f"Total Current Required: {self.total_current_required:.2f} A")
        
        if self.anodes:
            current_per_anode = self.total_current_required / len(self.anodes)
            print(f"Current per Anode: {current_per_anode:.2f} A")
            for anode in self.anodes:
                anode['current'] = current_per_anode

    def create_structure_mask(self, x, y):
        mask = np.zeros_like(x, dtype=bool)
        
        for member in self.structure_points:
            start = member['start']
            end = member['end']
            thickness = member['thickness']
            
            # Vector calculations for 2D
            direction = end - start
            length = np.linalg.norm(direction)
            direction = direction / length
            
            v = np.stack([
                x - start[0],
                y - start[1]
            ], axis=-1)
            
            projection = np.sum(v * direction, axis=-1)[..., np.newaxis] * direction
            distance = np.linalg.norm(v - projection, axis=-1)
            t = np.sum(v * direction, axis=-1)
            
            mask |= (distance <= thickness) & (t >= 0) & (t <= length)
        
        return mask

    def create_anode_mask(self, x, y):
        mask = np.zeros_like(x, dtype=bool)
        
        for anode in self.anodes:
            pos = anode['position']
            distance = np.sqrt(
                (x - pos[0])**2 +
                (y - pos[1])**2
            )
            mask |= (distance <= 1.0)  # Anode radius of 1.0 meters
        
        return mask

    def visualize_results(self):
        fig = go.Figure()
        
        # Add potential distribution contour with improved color scheme
        x = np.linspace(0, self.domain_size[0], self.grid_size[0])
        y = np.linspace(0, self.domain_size[1], self.grid_size[1])
        
        fig.add_trace(go.Contour(
            x=x, y=y,
            z=self.potential,
            colorscale=[
                [0.0, 'rgb(0,0,150)'],     # Dark blue for most negative
                [0.2, 'rgb(0,0,255)'],     # Blue
                [0.4, 'rgb(0,255,255)'],   # Cyan
                [0.6, 'rgb(0,255,0)'],     # Green
                [0.8, 'rgb(255,255,0)'],   # Yellow
                [1.0, 'rgb(255,0,0)']      # Red for most positive
            ],
            contours=dict(
                start=-1.1,
                end=-0.7,
                size=0.025,
                showlabels=True,
                labelfont=dict(size=12, color='black', family='Arial Bold'),
                labelformat='.3f'
            ),
            colorbar=dict(
                title=dict(
                    text='Potential (V vs. CSE)',
                    side='right',
                    font=dict(size=14, family='Arial Bold')
                ),
                thickness=25,
                len=0.8,
                tickformat='.2f',
                tickfont=dict(size=12)
            ),
            hoverongaps=False,
            connectgaps=True,  # Added this instead of smoothing
            ncontours=30      # Added for smoother contours
        ))
        
        # Add pipeline with improved style
        for member in self.structure_points:
            fig.add_trace(go.Scatter(
                x=[member['start'][0], member['end'][0]],
                y=[member['start'][1], member['end'][1]],
                mode='lines',
                line=dict(color='black', width=8),
                name='Pipeline',
                hoverinfo='name'
            ))
        
        # Add seabed with improved style
        fig.add_trace(go.Scatter(
            x=[0, self.domain_size[0]],
            y=[0, 0],
            mode='lines',
            line=dict(color='sienna', width=3, dash='solid'),
            name='Seabed',
            hoverinfo='name'
        ))
        
        # Add anodes with improved style
        anode_positions = np.array([anode['position'] for anode in self.anodes])
        fig.add_trace(go.Scatter(
            x=anode_positions[:, 0],
            y=anode_positions[:, 1],
            mode='markers',
            marker=dict(
                size=18,
                color='gold',
                line=dict(color='darkgoldenrod', width=2),
                symbol='diamond'
            ),
            name='ICCP Anodes',
            hoverinfo='name+text',
            hovertext=['Anode '+str(i+1) for i in range(len(anode_positions))]
        ))
        
        # Add current flow with improved style
        skip = 6  # Reduced skip for more arrows
        x_quiver = x[::skip]
        y_quiver = y[::skip]
        u_quiver = self.current_vectors[0][::skip, ::skip]
        v_quiver = self.current_vectors[1][::skip, ::skip]
        
        magnitude = np.sqrt(u_quiver**2 + v_quiver**2)
        u_norm = u_quiver / (magnitude + 1e-10)
        v_norm = v_quiver / (magnitude + 1e-10)
        
        fig.add_trace(go.Scatter(
            x=x_quiver.flatten(),
            y=y_quiver.flatten(),
            mode='markers',
            marker=dict(
                symbol='arrow',
                angle=np.arctan2(v_norm, u_norm).flatten() * 180 / np.pi,
                size=6,
                color='rgba(50,50,50,0.6)',
                line=dict(width=1, color='rgba(30,30,30,0.8)')
            ),
            name='Current Flow',
            hoverinfo='none'
        ))
        
        # Add annotation with improved style
        fig.add_annotation(
            text=f'<b>Simulation Parameters</b><br>' +
                 f'Coating Integrity: {self.coating_integrity*100:.1f}%<br>' +
                 f'Salinity: {self.salinity:.1f} PSU<br>' +
                 f'Required Current: {self.total_current_required:.2f} A',
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=2,
            borderpad=6,
            font=dict(size=12, family='Arial Bold')
        )
        
        fig.update_layout(
            title=dict(
                text='2D Subsea Pipeline Cathodic Protection Simulation',
                font=dict(size=20, family='Arial Bold'),
                y=0.98  # Moved up slightly
            ),
            xaxis_title=dict(
                text='Distance (m)', 
                font=dict(size=14, family='Arial'),
                standoff=20  # Added spacing between axis and title
            ),
            yaxis_title=dict(
                text='Height (m)', 
                font=dict(size=14, family='Arial'),
                standoff=20  # Added spacing between axis and title
            ),
            showlegend=True,
            legend=dict(
                x=0.99,
                y=0.99,
                xanchor='right',  # Align to right
                yanchor='top',    # Align to top
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12),
                itemsizing='constant'  # Consistent legend item sizes
            ),
            width=1200,
            height=800,  # Increased height for better spacing
            xaxis=dict(
                range=[0, self.domain_size[0]], 
                showgrid=True, 
                gridwidth=1, 
                gridcolor='lightgray',
                title_standoff=25  # Added spacing
            ),
            yaxis=dict(
                range=[0, self.domain_size[1]], 
                showgrid=True, 
                gridwidth=1, 
                gridcolor='lightgray',
                title_standoff=25  # Added spacing
            ),
            margin=dict(t=100, b=80, l=80, r=80),  # Adjusted margins
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig

    def calculate_current_density(self):
        dx = self.domain_size[0] / (self.grid_size[0] - 1)
        dy = self.domain_size[1] / (self.grid_size[1] - 1)
        
        Ex, Ey = np.gradient(-self.potential, dx, dy)
        conductivity = self.materials['seawater'].conductivity
        
        self.current_vectors = (
            Ex * conductivity,
            Ey * conductivity
        )
        
        self.current_density = np.sqrt(
            self.current_vectors[0]**2 +
            self.current_vectors[1]**2
        )

    def calculate_potential_distribution(self):
        nx, ny = self.grid_size
        N = nx * ny
        
        # Create sparse matrix for 2D Laplace equation
        row, col, data = [], [], []
        b = np.zeros(N)
        
        # Setup mesh
        x, y = np.meshgrid(
            np.linspace(0, self.domain_size[0], nx),
            np.linspace(0, self.domain_size[1], ny)
        )
        
        # Create structure and anode masks
        structure_mask = self.create_structure_mask(x, y)
        anode_mask = np.zeros_like(x, dtype=bool)
        
        # Create individual anode masks and apply currents
        for anode in self.anodes:
            pos = anode['position']
            distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            current_mask = distance <= 1.0
            anode_mask |= current_mask
            
            # Apply current contribution to boundary conditions
            b[current_mask.flatten()] += anode['current'] / np.sum(current_mask)
        
        for i in range(nx):
            for j in range(ny):
                index = i + j * nx
                
                if structure_mask[i,j]:
                    row.append(index)
                    col.append(index)
                    data.append(1.0)
                    b[index] = self.materials['steel'].potential
                elif anode_mask[i,j]:
                    row.append(index)
                    col.append(index)
                    data.append(1.0)
                    # Potential at anode locations is already set in b array
                else:
                    row.append(index)
                    col.append(index)
                    data.append(-4.0)
                    
                    for offset in [1, -1, nx, -nx]:
                        if 0 <= index + offset < N:
                            row.append(index)
                            col.append(index + offset)
                            data.append(1.0)
        
        A = csr_matrix((data, (row, col)), shape=(N, N))
        solution = linalg.spsolve(A, b)
        self.potential = solution.reshape(self.grid_size)
        self.calculate_current_density()

# Update the run function default parameter
def run_2d_simulation(coating_integrity=0.85, salinity=50.0, water_agitation=0.75):
    sim = OffshoreICCPSimulator2D()
    sim.coating_integrity = coating_integrity
    sim.salinity = salinity
    sim.water_agitation = water_agitation
    sim.setup_materials()
    sim.add_structure()
    sim.add_iccp_anodes(num_anodes=4)
    sim.calculate_potential_distribution()
    fig = sim.visualize_results()
    fig.show()
    return sim

if __name__ == "__main__":
    simulation = run_2d_simulation(coating_integrity=0.85, salinity=50.0, water_agitation=0.75)