"""
Particle Filter for Fuel Level Estimation - Complete Dataset
Processes ALL measurements with adaptive rate learning
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ParticleFilter:
    """Particle Filter for fuel estimation"""
    
    def __init__(self, num_particles=10, initial_fuel=25, sensor_noise=0.5,
                 motion_noise=0.1, resample_noise=0.05):
        self.particles = np.random.normal(initial_fuel, sensor_noise, num_particles)
        self.weights = np.ones(num_particles) / num_particles
        self.num_particles = num_particles
        
        self.sensor_noise = sensor_noise
        self.motion_noise = motion_noise
        self.resample_noise = resample_noise
        
        self.estimates = [initial_fuel]
        self.uncertainties = [np.std(self.particles)]
        
    def sampling(self, consumption):
        noise = np.random.normal(0, self.motion_noise, self.num_particles)
        self.particles = self.particles - consumption + noise
        self.particles = np.maximum(self.particles, 0)
    
    def weighting(self, measurement):
        errors = self.particles - measurement
        likelihoods = np.exp(-(errors ** 2) / (2 * self.sensor_noise ** 2))
        likelihoods[~np.isfinite(likelihoods)] = 1e-300  # Handle underflow
        weight_sum = np.sum(likelihoods)
        if weight_sum <= 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = likelihoods / weight_sum
        self.weights[~np.isfinite(self.weights)] = 1.0 / self.num_particles
    
    def roulette_wheel_selection(self):
        cumsum = np.cumsum(self.weights)
        r = np.random.uniform(0, 1)
        index = np.searchsorted(cumsum, r)
        return min(index, self.num_particles - 1)
    
    def resampling(self):
        """Optimized resampling with vectorization"""
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices] + np.random.normal(0, self.resample_noise, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def step(self, consumption, measurement):
        self.sampling(consumption)
        self.weighting(measurement)
        self.resampling()
        
        estimate = np.mean(self.particles)
        uncertainty = np.std(self.particles)
        
        self.estimates.append(estimate)
        self.uncertainties.append(uncertainty)
        
        return estimate, uncertainty


def load_data(csv_path=None):
    """Load complete CSV data"""
    fuel_col = 'ros_main__generator_controller__hatz_info__fuel_level__value'
    speed_col = 'ros_main__inverse_kinematics__cmd__data__speed'
    
    if csv_path is None:
        csv_path = '/mnt/user-data/uploads/fuel_cmd.csv'
    
    df = pd.read_csv(csv_path)
    df_clean = pd.DataFrame({
        'fuel_level': df[fuel_col].fillna(0),
        'speed': df[speed_col].fillna(0)
    })
    
    df_clean = df_clean[df_clean['fuel_level'] > 0].reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_clean):,} measurements")
    print(f"  Fuel: {df_clean['fuel_level'].min():.2f} - {df_clean['fuel_level'].max():.2f} L")
    print(f"  Speed: {df_clean['speed'].min():.2f} - {df_clean['speed'].max():.2f} m/s")
    
    return df_clean


def run_particle_filter(df, num_particles=10):
    """Run particle filter"""
    
    print(f"\n{'='*60}")
    print("PARTICLE FILTER")
    print(f"{'='*60}")
    print(f"Particles: {num_particles}")
    print(f"Measurements: {len(df):,}\n")
    
    pf = ParticleFilter(num_particles=num_particles, initial_fuel=df.iloc[0]['fuel_level'])
    
    estimates = [df.iloc[0]['fuel_level']]
    refuel_events = []
    
    # Initialize adaptive rates
    fuel_rate_moving = 0.02
    fuel_rate_idle = 0.005
    adaptation_alpha = 0.05
    
    for i in range(1, len(df)):
        curr_fuel = df.iloc[i]['fuel_level']
        curr_speed = df.iloc[i]['speed']
        prev_fuel = df.iloc[i-1]['fuel_level']
        prev_speed = df.iloc[i-1]['speed']
        
        # Simple refueling detection: large upward jump (>15L)
        if curr_fuel - prev_fuel > 15.0:
            print(f"  [Step {i:,}] REFUEL: {prev_fuel:.2f}L → {curr_fuel:.2f}L")
            refuel_events.append(i)
            pf.particles = np.random.normal(curr_fuel, pf.sensor_noise, pf.num_particles)
            pf.weights = np.ones(pf.num_particles) / pf.num_particles
        
        # Motion model: use adaptive learned rates (speed-based, sensor-independent)
        if abs(curr_speed) > 0.15:
            consumption = fuel_rate_moving
        else:
            consumption = fuel_rate_idle
        
        # SWR step: Sampling, Weighting, Resampling
        estimate, uncertainty = pf.step(consumption, curr_fuel)
        estimates.append(estimate)
        
        # Adaptation (outside SWR loop) - learn consumption from observations
        if i not in refuel_events:
            observed = max(prev_fuel - curr_fuel, 0)
            # Only learn from small continuous changes (filter out discrete sensor noise)
            if 0 < observed < 1.0:
                if abs(prev_speed) > 0.15:
                    fuel_rate_moving = (1 - adaptation_alpha) * fuel_rate_moving + \
                                      adaptation_alpha * observed
                else:
                    fuel_rate_idle = (1 - adaptation_alpha) * fuel_rate_idle + \
                                    adaptation_alpha * observed
        
        # Progress indicator (every 20%)
        if (i + 1) % max(1, len(df) // 5) == 0:
            progress = 100 * (i + 1) / len(df)
            print(f"  {progress:3.0f}% - Step {i:,} | Est: {estimate:.2f}L")
    
    print(f"\n✓ Completed!")
    print(f"  Refueling events: {len(refuel_events)}")
    print(f"  Final rates: moving={fuel_rate_moving:.5f}, idle={fuel_rate_idle:.5f} L/min")
    
    return {
        'df': df,
        'estimates': np.array(estimates),
        'uncertainties': np.array(pf.uncertainties),
        'refuel_events': refuel_events
    }


def plot_results(results, output_file='particle_filter_complete.html'):
    """Create 2-row plot: Fuel levels + Error"""
    
    df = results['df']
    estimates = results['estimates']
    uncertainties = results['uncertainties']
    refuel_events = results['refuel_events']
    
    time = np.arange(len(df))
    error = df['fuel_level'].values - estimates
    
    # Create 2-row subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Fuel Levels (raw & estimates)", "Estimation Error"),
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35]
    )
    
    # ROW 1: FUEL LEVELS
    fig.add_trace(
        go.Scatter(
            x=time, y=df['fuel_level'].values,
            mode='lines',
            name='Fuel (raw)',
            line=dict(color='lightgray', width=1),
            hovertemplate='Step: %{x:,}<br>Sensor: %{y:.3f}L'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time, y=estimates,
            mode='lines',
            name='Particle Filter',
            line=dict(color='orange', width=2),
            hovertemplate='Step: %{x:,}<br>Estimate: %{y:.3f}L'
        ),
        row=1, col=1
    )
    
    # Shade refueling events
    for refuel_idx in refuel_events:
        fig.add_vline(x=refuel_idx, line_dash="dash", line_color="red",
                     row=1, col=1, opacity=0.4)
    
    # ROW 2: ERROR
    fig.add_trace(
        go.Scatter(
            x=time, y=error,
            mode='lines',
            name='Error',
            line=dict(color='firebrick', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(178, 34, 34, 0.2)',
            hovertemplate='Step: %{x:,}<br>Error: %{y:.3f}L'
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1, opacity=0.3)
    
    # Labels
    fig.update_yaxes(title_text="Fuel (L)", row=1, col=1)
    fig.update_yaxes(title_text="Error (L)", row=2, col=1)
    fig.update_xaxes(title_text="Measurement Step", row=2, col=1)
    
    rmse = np.sqrt(np.mean(error**2))
    mean_error = np.mean(error)
    std_error = np.std(error)
    
    fig.update_layout(
        title={
            'text': f"Particle Filter - Complete Dataset ({len(df):,} measurements)<br>" +
                   f"<sub>RMSE: {rmse:.4f}L | Mean Error: {mean_error:.4f}L | " +
                   f"Std: {std_error:.4f}L | Refueling Events: {len(refuel_events)}</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        template='plotly_white',
        hovermode='x unified',
        height=900,
        legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top')
    )
    
    fig.write_html(output_file)
    print(f"\n✓ Plot saved: {output_file}")
    print(f"  RMSE: {rmse:.4f}L")
    print(f"  Mean Error: {mean_error:.4f}L")
    print(f"  Std Dev: {std_error:.4f}L")


def main():
    print("\n" + "="*60)
    print("PARTICLE FILTER")
    print("="*60)
    
    # Load complete data
    df = load_data('data/fuel_cmd.csv')
    
    results = run_particle_filter(df, num_particles=10)
    
    # Plot results
    plot_results(results, 'particle_filter_complete.html')
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()