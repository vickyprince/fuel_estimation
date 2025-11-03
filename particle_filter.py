import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ParticleFilterSimplified:
    """
    Simplified Particle Filter with working refuel detection
    Based on successful pattern from original but fixed
    """
    
    def __init__(self, num_particles=2000, initial_fuel=25.0,
                 process_noise=0.15, measurement_noise=0.5,
                 learning_rate=0.1):
        
        self.num_particles = num_particles
        self.Q = process_noise
        self.R = measurement_noise
        self.rate_alpha = learning_rate
        
        # Initialize particles
        self.particles = np.random.uniform(0, 35, num_particles)
        self.particles = np.maximum(self.particles, 0)
        self.weights = np.ones(num_particles) / num_particles

        self.last_timestamp = None
        self.rate_moving = 0.017 / 60.0    
        self.rate_idle = 0.017 / 60.0      
        
        # History
        self.estimates = []
        self.rates = []
        self.particles_history = []
        
        # Simplified refuel detection
        self.fuel_history = []
        self.speed_history = []
        self.history_window = 20
        self.last_refuel_step = -50
        self.refuel_cooldown = 30
        
    def detect_refuel_simple(self, prev_fuel, curr_fuel, curr_speed, step_num):
        """
        Simplified but effective refuel detection
        Focus on what works: large jumps when stopped
        """
        # Update history
        self.fuel_history.append(curr_fuel)
        self.speed_history.append(curr_speed)
        
        if len(self.fuel_history) > self.history_window:
            self.fuel_history.pop(0)
            self.speed_history.pop(0)
        
        # Basic cooldown
        # if step_num - self.last_refuel_step < self.refuel_cooldown:
        #     return False, 0.0
        
        # Need some history
        if len(self.fuel_history) < 5:
            return False, 0.0
        
        # Calculate fuel jump
        fuel_jump = curr_fuel - prev_fuel
        
        # Main detection criteria
        # 1. Significant positive jump
        if fuel_jump < 5.0:  # Minimum 5L jump
            return False, 0.0
        
        # 2. Robot should be relatively stopped
        if abs(curr_speed) > 0.15:  # Relaxed threshold
            return False, 0.0
        
        # 3. Check recent speed history - was it stopped recently?
        if len(self.speed_history) >= 5:
            recent_speeds = self.speed_history[-5:]
            stopped_count = sum(1 for s in recent_speeds if abs(s) < 0.15)
            if stopped_count < 3:  # At least 3/5 samples stopped
                return False, 0.0
        
        # 4. Fuel was generally stable or decreasing before
        if len(self.fuel_history) >= 5:
            # Check if fuel was not already jumping up
            prev_changes = [self.fuel_history[i] - self.fuel_history[i-1] 
                           for i in range(len(self.fuel_history)-4, len(self.fuel_history)-1)]
            
            # If recent changes were also large positive, might be noise
            if any(change > 3.0 for change in prev_changes):
                return False, 0.0
        
        # 5. Simple anti-bounce: check for recent large negative
        if len(self.fuel_history) >= 3:
            recent_drop = self.fuel_history[-2] - self.fuel_history[-3]
            if recent_drop < -3.0:  # Large drop just before
                return False, 0.0  # Likely a bounce
        
        # Calculate confidence based on jump size and stop duration
        jump_confidence = min(1.0, fuel_jump / 15.0)  # 15L jump = full confidence
        stop_confidence = stopped_count / 5.0 if len(self.speed_history) >= 5 else 0.5
        confidence = 0.6 * jump_confidence + 0.4 * stop_confidence
        
        # Accept refuel
        self.last_refuel_step = step_num
        print(f"✓ Refuel detected at step {step_num}: "
              f"{prev_fuel:.2f}L → {curr_fuel:.2f}L "
              f"(jump: {fuel_jump:.2f}L, confidence: {confidence:.2f})")
        
        return True, confidence
    
    def predict(self, consumption_rate_l_per_sec, dt):
        """Predict with adaptive noise"""
        consumption = consumption_rate_l_per_sec * dt
        self.particles -= consumption
        self.particles += np.random.normal(0, self.Q, self.num_particles)
        self.particles = np.maximum(self.particles, 0)
        
    def update(self, measurement):
        """Update weights based on measurement"""
        errors = self.particles - measurement
        logw = -0.5 * (errors**2) / (self.R**2)
        logw -= np.max(logw)
        w = np.exp(logw)
        w = np.clip(w, 1e-300, None)
        self.weights = w / np.sum(w)

    # def resample_ess_systematic(self):
    #     """Resample if effective sample size is low"""
    #     ess = 1.0 / np.sum(self.weights ** 2)
    #     if ess < 0.5 * self.num_particles:
    #         cdf = np.cumsum(self.weights)
    #         u = (np.arange(self.num_particles) + np.random.uniform()) / self.num_particles
    #         indices = np.searchsorted(cdf, u)
    #         self.particles = self.particles[indices].copy()
    #         sigma = 0.05 * max(1.0, np.std(self.particles))
    #         self.particles += np.random.normal(0, sigma, self.particles.shape)
    #         self.particles = np.maximum(self.particles, 0)
    #         self.weights.fill(1.0/self.num_particles)
    #         return True, ess
    #     return False, ess
    
    def resample_ess_systematic(self):
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < 0.5 * self.num_particles:
            cdf = np.cumsum(self.weights)
            u = (np.arange(self.num_particles) + np.random.uniform()) / self.num_particles
            indices = np.searchsorted(cdf, u)
            self.particles = self.particles[indices].copy()
            # Roughening proportional to state span / N^(1/d)
            span = np.percentile(self.particles, 95) - np.percentile(self.particles, 5)
            roughen_sigma = 0.2 * span / (self.num_particles ** (1/1))  # d=1
            self.particles += np.random.normal(0, max(0.5, roughen_sigma), self.num_particles)
            self.particles = np.maximum(self.particles, 0)
            self.weights.fill(1.0/self.num_particles)
            return True, ess
        return False, ess
    
    def adapt_rates(self, prev_fuel, curr_fuel, is_refuel, speed, dt):
        """Adaptive learning of consumption rates"""
        if is_refuel or dt <= 0:
            return
            
        observed = max(prev_fuel - curr_fuel, 0)
        
        if 0.001 < observed < 0.5:
            observed_rate_l_per_sec = observed / dt
            actual_alpha = min(self.rate_alpha, 0.01)
            observed_rate_l_per_sec = np.clip(observed_rate_l_per_sec, 0, 0.01)
            
            if abs(speed) > 0.15:
                new_rate = (1 - actual_alpha) * self.rate_moving + actual_alpha * observed_rate_l_per_sec
                if 0 < new_rate < 0.01:
                    self.rate_moving = new_rate
            else:
                new_rate = (1 - actual_alpha) * self.rate_idle + actual_alpha * observed_rate_l_per_sec
                if 0 < new_rate < 0.01:
                    self.rate_idle = new_rate
    
    def step(self, prev_fuel, curr_fuel, curr_speed, curr_timestamp, step_num=0):
        """
        Full step with simplified detection
        """
        is_refuel, confidence = self.detect_refuel_simple(
            prev_fuel, curr_fuel, curr_speed, step_num
        )
        
        # Handle refuel
        if is_refuel:
            # Reset particles based on confidence
            reset_ratio = 0.3 + 0.4 * confidence
            n_reset = int(self.num_particles * reset_ratio)
            reset_indices = np.random.choice(self.num_particles, n_reset, replace=False)

            self.particles[reset_indices] = np.random.normal(curr_fuel, 0.5, n_reset)
            self.particles = np.maximum(self.particles, 0)
            self.weights = np.ones(self.num_particles) / self.num_particles

        # Calculate dt
        if self.last_timestamp is not None:
            if isinstance(curr_timestamp, (int, float)):
                dt = curr_timestamp - self.last_timestamp
            else:
                dt = (curr_timestamp - self.last_timestamp).total_seconds()
        else:
            dt = 1.0

        self.last_timestamp = curr_timestamp
        
        # Select consumption rate
        consumption = self.rate_moving if abs(curr_speed) > 0.15 else self.rate_idle
        
        # Learn rates
        self.adapt_rates(prev_fuel, curr_fuel, is_refuel, curr_speed, dt)
        
        # Particle filter steps
        self.predict(consumption, dt)
        self.update(curr_fuel)
        
        # Store particle statistics
        p5 = np.percentile(self.particles, 5)
        p25 = np.percentile(self.particles, 25)
        p50 = np.percentile(self.particles, 50)
        p75 = np.percentile(self.particles, 75)
        p95 = np.percentile(self.particles, 95)

        self.particles_history.append({
            'p5': p5, 'p25': p25, 'p50': p50, 'p75': p75, 'p95': p95,
            'is_refuel': is_refuel,
            'confidence': confidence if is_refuel else 0.0
        })
        
        # Resample if needed
        resampled, ess = self.resample_ess_systematic()
        
        # Compute estimate
        fuel_est = np.average(self.particles, weights=self.weights)
        
        # Store history
        self.estimates.append(fuel_est)
        self.rates.append((self.rate_moving + self.rate_idle) / 2)
        
        return fuel_est, is_refuel, confidence


def load_data(csv_path='data/fuel_cmd.csv', nrows=None):
    """Load and clean CSV data"""
    fuel_col = 'ros_main__generator_controller__hatz_info__fuel_level__value'
    speed_col = 'ros_main__inverse_kinematics__cmd__data__speed'
    
    df = pd.read_csv(csv_path, nrows=nrows)
    df_clean = pd.DataFrame({
        'fuel_level': df[fuel_col].fillna(0),
        'speed': df[speed_col].fillna(0),
        'timestamp': df['timestamp']
    })
    
    df_clean = df_clean[df_clean['fuel_level'] > 0].reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_clean):,} measurements")
    print(f"  Fuel: {df_clean['fuel_level'].min():.2f} - {df_clean['fuel_level'].max():.2f} L")
    print(f"  Speed: {df_clean['speed'].min():.2f} - {df_clean['speed'].max():.2f} m/s")
    
    return df_clean


def run_filter(df, num_particles=500, learning_rate=0.01, 
               process_noise=0.4, measurement_noise=0.8):
    """Run the simplified particle filter"""
    
    pf = ParticleFilterSimplified(
        num_particles=num_particles,
        initial_fuel=df.iloc[0]['fuel_level'],
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        learning_rate=learning_rate
    )
    
    estimates = []
    refuel_events = []
    refuel_confidences = []
    refuel_count = 0
    
    print("\nProcessing measurements...")
    
    for i in range(1, len(df)):
        curr_fuel = df.iloc[i]['fuel_level']
        curr_speed = df.iloc[i]['speed']
        prev_fuel = df.iloc[i-1]['fuel_level']
        curr_timestamp = df.iloc[i]['timestamp']
        
        # Run step
        fuel_est, is_refuel, confidence = pf.step(
            prev_fuel, curr_fuel, curr_speed, curr_timestamp, step_num=i
        )
        
        estimates.append(fuel_est)
        
        # Track refueling
        if is_refuel:
            refuel_count += 1
            refuel_events.append(i)
            refuel_confidences.append(confidence)
        
        # Progress updates
        if (i + 1) % (len(df) // 5) == 0:
            pct = 100 * (i + 1) / len(df)
            error = abs(fuel_est - curr_fuel)
            print(f"{pct:3.0f}% - Est: {fuel_est:.2f}L | "
                  f"Sensor: {curr_fuel:.2f}L | Error: {error:.2f}L | "
                  f"Refuels: {refuel_count}")
    
    # Calculate metrics
    error = df['fuel_level'].values[1:] - estimates
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  RMSE: {rmse:.4f}L")
    print(f"  MAE:  {mae:.4f}L")
    print(f"  Refueling events detected: {refuel_count}")
    if refuel_confidences:
        print(f"  Average confidence: {np.mean(refuel_confidences):.2f}")
    print(f"  Final rates - Moving: {pf.rate_moving*3600:.6f} L/hour")
    print(f"  Final rates - Idle: {pf.rate_idle*3600:.6f} L/hour")
    print(f"{'='*70}\n")
    
    return {
        'df': df[1:].reset_index(drop=True),
        'estimates': np.array(estimates),
        'rates': np.array(pf.rates),
        'rmse': rmse,
        'mae': mae,
        'particles_history': pf.particles_history,
        'refuel_events': refuel_events,
        'refuel_confidences': refuel_confidences,
        'refuel_count': refuel_count
    }


def plot_results(results, output_file='particle_filter_simplified.html'):
    """Create visualization"""
    df = results['df']
    est = results['estimates']
    rates = results['rates']
    rmse = results['rmse']
    mae = results['mae']
    refuel_count = results.get('refuel_count', len(results['refuel_events']))
    
    time = np.arange(len(df))
    error = df['fuel_level'].values - est
    
    # Create subplot
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"Fuel Estimation (Refuels Detected: {refuel_count})",
            "Learned Consumption Rate",
            "Estimation Error"
        ),
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.1
    )
    
    # Extract particle percentiles
    particles_hist = results['particles_history']
    p5_arr = np.array([p['p5'] for p in particles_hist])
    p25_arr = np.array([p['p25'] for p in particles_hist])
    p75_arr = np.array([p['p75'] for p in particles_hist])
    p95_arr = np.array([p['p95'] for p in particles_hist])

    # Row 1: Fuel tracking
    fig.add_trace(
        go.Scatter(x=time, y=df['fuel_level'], mode='lines',
                name='Sensor', line=dict(color='gray', width=1.5)),
        row=1, col=1
    )

    # Particle bands
    fig.add_trace(
        go.Scatter(x=time, y=p95_arr, fill=None, mode='lines',
                line_color='rgba(0,0,0,0)', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=p5_arr, fill='tonexty', mode='lines',
                line_color='rgba(0,0,0,0)', fillcolor='rgba(100,149,237,0.2)',
                name='Particle Range (5-95%)'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=time, y=p75_arr, fill=None, mode='lines',
                line_color='rgba(0,0,0,0)', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=p25_arr, fill='tonexty', mode='lines',
                line_color='rgba(0,0,0,0)', fillcolor='rgba(30,144,255,0.3)',
                name='Particle IQR (25-75%)'),
        row=1, col=1
    )
    
    # PF Estimate
    fig.add_trace(
        go.Scatter(x=time, y=est, mode='lines',
                name='PF Estimate', line=dict(color='orange', width=2.5)),
        row=1, col=1
    )
    
    # Mark refuels
    if results['refuel_events']:
        refuel_indices = [idx-1 for idx in results['refuel_events'] if idx-1 < len(time)]
        refuel_times = [time[idx] for idx in refuel_indices]
        refuel_values = [df['fuel_level'].iloc[idx] for idx in refuel_indices]
        
        fig.add_trace(
            go.Scatter(x=refuel_times, y=refuel_values, mode='markers',
                    name='Refuel Events',
                    marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )
        
    # Row 2: Rates
    fig.add_trace(
        go.Scatter(x=time, y=rates, mode='lines', name='Learned Rate',
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    # Row 3: Error
    fig.add_trace(
        go.Scatter(x=time, y=error, mode='lines', name='Error',
                  line=dict(color='red', width=1.5), fill='tozeroy',
                  fillcolor='rgba(255,0,0,0.15)'),
        row=3, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1, opacity=0.3)
    
    # Layout
    fig.update_layout(
        title=f"Simplified PF | RMSE: {rmse:.3f}L | MAE: {mae:.3f}L | Refuels: {refuel_count}",
        height=1000, 
        template='plotly_white', 
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    
    fig.update_yaxes(title_text="Fuel (L)", row=1, col=1)
    fig.update_yaxes(title_text="Rate (L/step)", row=2, col=1)
    fig.update_yaxes(title_text="Error (L)", row=3, col=1)
    fig.update_xaxes(title_text="Measurement Step", row=3, col=1)
    
    fig.write_html(output_file)
    print(f"✓ Visualization saved: {output_file}")


def main():
    """Main with simplified configuration"""
    print("\n" + "="*70)
    print("PARTICLE FILTER")
    print("="*70)

    nrows = sum(1 for _ in open('data/fuel_cmd.csv')) // 2

    df = load_data(nrows=nrows)
    
    print(f"Data: {len(df):,} measurements")
    
    # Run with simplified configuration
    results = run_filter(
        df, 
        num_particles=10,      
        learning_rate=0.01,     
        process_noise=0.1,      
        measurement_noise=1.5   
    )
    
    # Create visualization
    plot_results(results, 'particle_filter_simplified.html')
    
    print("✓ Complete!\n")


if __name__ == '__main__':
    main()