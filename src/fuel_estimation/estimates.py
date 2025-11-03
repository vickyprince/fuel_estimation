"""
Particle Filter for Fuel Level Estimation - Simplified
Shows: Fuel (sensor vs estimate) + Estimation Error
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ParticleFilter:
    """Particle Filter for fuel estimation"""
    
    def __init__(self, num_particles=1000, initial_fuel=25, sensor_noise=0.5,  # Increased from 0.15
                 motion_noise=0.1, resample_noise=0.05):  # Increased noise levels
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
        self.weights = likelihoods / np.sum(likelihoods)
    
    def roulette_wheel_selection(self):
        cumsum = np.cumsum(self.weights)
        r = np.random.uniform(0, 1)
        index = np.searchsorted(cumsum, r)
        return min(index, self.num_particles - 1)
    
    def resampling(self):
        new_particles = []
        for _ in range(self.num_particles):
            idx = self.roulette_wheel_selection()
            new_particle = self.particles[idx] + np.random.normal(0, self.resample_noise)
            new_particles.append(new_particle)
        self.particles = np.array(new_particles)
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
    """Load CSV data"""
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
    
    print(f"✓ Loaded {len(df_clean)} measurements")
    print(f"  Fuel: {df_clean['fuel_level'].min():.2f} - {df_clean['fuel_level'].max():.2f} L")
    
    return df_clean


def calculate_consumption(speed, fuel_rate_moving, fuel_rate_idle, speed_threshold=0.15):
    """
    Calculate consumption based on speed and learned rates
    
    Motion model uses learned consumption rates (updated outside SWR loop)
    Speed only enters here to select which rate to use
    """
    if abs(speed) > speed_threshold:
        consumption = fuel_rate_moving  # Learned rate when moving
    else:
        consumption = fuel_rate_idle    # Learned rate when idle
    
    return consumption


def detect_refueling_simple(prev_sensor_fuel, curr_sensor_fuel, fuel_threshold=15.0):
    """
    Simple refueling detection: detect large upward JUMPS in sensor data
    
    Logic:
    - Refueling causes fuel to jump UP significantly
    - If sensor fuel increases by > threshold, it's likely refueling
    - We only reset particles when this happens (they can't explain the jump)
    
    This is reactive: we don't predict refueling, we react to it when it happens
    """
    fuel_increase = curr_sensor_fuel - prev_sensor_fuel
    return fuel_increase > fuel_threshold


def run_particle_filter(df, num_particles=1000, max_samples=10000):
    """Run particle filter with adaptive rate learning"""
    
    if len(df) > max_samples:
        df = df.iloc[:max_samples].reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print("PARTICLE FILTER - FUEL ESTIMATION")
    print(f"{'='*60}")
    print(f"Particles: {num_particles}")
    print(f"Measurements: {len(df)}\n")
    
    pf = ParticleFilter(num_particles=num_particles, initial_fuel=df.iloc[0]['fuel_level'])
    
    estimates = [df.iloc[0]['fuel_level']]
    refuel_events = []
    
    # Initialize learned rates (will adapt over time)
    fuel_rate_moving = 0.02   # Initial guess (L/min)
    fuel_rate_idle = 0.005    # Initial guess (L/min)
    adaptation_alpha = 0.05   # Learning rate (0-1)
    
    rate_history = [(fuel_rate_moving, fuel_rate_idle)]
    
    for i in range(1, len(df)):
        curr_fuel = df.iloc[i]['fuel_level']
        curr_speed = df.iloc[i]['speed']
        prev_fuel = df.iloc[i-1]['fuel_level']
        prev_speed = df.iloc[i-1]['speed']
        
        # Simple refueling detection: check for large upward jump in sensor
        if detect_refueling_simple(prev_fuel, curr_fuel, fuel_threshold=15.0):
            print(f"  [Step {i}] REFUEL: {prev_fuel:.2f}L → {curr_fuel:.2f}L (jump: +{curr_fuel-prev_fuel:.2f}L)")
            refuel_events.append(i)
            
            # Reset particles to new sensor reading
            pf.particles = np.random.normal(curr_fuel, pf.sensor_noise, pf.num_particles)
            pf.weights = np.ones(pf.num_particles) / pf.num_particles
        
        # Motion model: consumption based on speed and learned rates (sensor independent)
        consumption = calculate_consumption(curr_speed, fuel_rate_moving, fuel_rate_idle)
        
        # Run SWR step with measurement
        estimate, uncertainty = pf.step(consumption, curr_fuel)
        estimates.append(estimate)
        
        # ===== ADAPTATION STEP (Outside SWR loop) =====
        # Learn consumption rate from sensor observation (only if not refueling)
        if i not in refuel_events:
            observed_consumption = max(prev_fuel - curr_fuel, 0)
            
            # IMPORTANT: Only learn from SMALL changes (< 1L)
            # Large changes are discrete sensor noise or partial refueling
            # We want to learn real consumption, not sensor artifacts
            if 0 < observed_consumption < 1.0:
                # Update rates based on speed (exponential moving average)
                if abs(prev_speed) > 0.15:  # Was moving
                    fuel_rate_moving = (1 - adaptation_alpha) * fuel_rate_moving + \
                                      adaptation_alpha * observed_consumption
                else:  # Was idle
                    fuel_rate_idle = (1 - adaptation_alpha) * fuel_rate_idle + \
                                    adaptation_alpha * observed_consumption
        
        rate_history.append((fuel_rate_moving, fuel_rate_idle))
        
        if (i + 1) % (len(df) // 5) == 0:
            progress = 100 * (i + 1) / len(df)
            print(f"  {progress:.0f}% - Est: {estimate:.2f}L | Rates: moving={fuel_rate_moving:.4f}, idle={fuel_rate_idle:.4f} L/min")
    
    print(f"\n✓ Completed | Refueling events detected: {len(refuel_events)}")
    print(f"Final rates: moving={fuel_rate_moving:.4f} L/min, idle={fuel_rate_idle:.4f} L/min")
    
    return {
        'df': df,
        'estimates': np.array(estimates),
        'uncertainties': np.array(pf.uncertainties),
        'refuel_events': refuel_events,
        'rates': rate_history
    }


def plot_results(results, output_file='particle_filter_results.html'):
    """Create simplified 2-row plot: Fuel + Error"""
    
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
    # Raw sensor data (gray)
    fig.add_trace(
        go.Scatter(
            x=time, y=df['fuel_level'].values,
            mode='lines',
            name='Fuel (raw)',
            line=dict(color='lightgray', width=1),
            hovertemplate='Step: %{x}<br>Sensor: %{y:.3f}L'
        ),
        row=1, col=1
    )
    
    # Particle filter estimate (orange)
    fig.add_trace(
        go.Scatter(
            x=time, y=estimates,
            mode='lines',
            name='Particle Filter',
            line=dict(color='orange', width=2.5),
            hovertemplate='Step: %{x}<br>Estimate: %{y:.3f}L'
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
            line=dict(color='firebrick', width=2),
            fill='tozeroy',
            fillcolor='rgba(178, 34, 34, 0.2)',
            hovertemplate='Step: %{x}<br>Error: %{y:.3f}L'
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1, opacity=0.3)
    
    # Labels
    fig.update_yaxes(title_text="Fuel (L)", row=1, col=1)
    fig.update_yaxes(title_text="Error (L)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    rmse = np.sqrt(np.mean(error**2))
    
    fig.update_layout(
        title={
            'text': f"Particle Filter - Fuel Estimation<br><sub>RMSE: {rmse:.3f}L | Mean Error: {np.mean(error):.3f}L | Refueling: {len(refuel_events)}</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        template='plotly_white',
        hovermode='x unified',
        height=900,
        legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top')
    )
    
    fig.write_html(output_file)
    print(f"✓ Plot saved: {output_file}")


def main():
    print("\n" + "="*60)
    print("PARTICLE FILTER - SIMPLIFIED VISUALIZATION")
    print("="*60 + "\n")
    
    # Load data
    df = load_data('/mnt/user-data/uploads/fuel_cmd.csv')
    
    # Run filter
    results = run_particle_filter(df, num_particles=1000, max_samples=10000)
    
    # Plot
    plot_results(results, '/mnt/user-data/outputs/particle_filter_simple.html')
    print("\n✓ Done!")


if __name__ == '__main__':
    main()