import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass


@dataclass
class ParticleFilterConfig:
    """Configuration for 2D Particle Filter"""
    num_particles: int = 2000
    # Process noise (for motion model)
    fuel_process_noise: float = 0.1  # Noise on fuel state
    rate_process_noise: float = 0.0001  # Noise on rate state
    # Measurement noise
    measurement_noise: float = 0.5  # Observation noise on fuel
    # Resampling
    ess_threshold: float = 0.5  # Resample when ESS drops below this fraction of N


class ParticleFilter2D:
    """
    2D Particle Filter with state = [fuel, consumption_rate]
    
    This filter explicitly models both fuel level and consumption rate as coupled states.
    Unlike the 1D approach which applies smoothing post-hoc, this 2D formulation allows
    the filter to explore the joint distribution and naturally estimate uncertainty.
    
    Key advantages over 1D:
    - Rate is a true state variable, not a separate adaptation
    - Motion model couples fuel and rate naturally
    - Particle distribution captures correlation between fuel and rate
    - No need for EMA smoothing - posterior distribution is the estimate
    """
    
    def __init__(self, config: ParticleFilterConfig):
        self.config = config
        self.num_particles = config.num_particles
        
        # State particles: shape (num_particles, 2)
        # Column 0: fuel level (liters)
        # Column 1: consumption rate (liters/second)
        self.particles = np.zeros((config.num_particles, 2))
        self.weights = np.ones(config.num_particles) / config.num_particles
        
        # History tracking
        self.estimates_mean = []  # Weighted mean estimate
        self.estimates_cov = []   # Covariance of posterior
        self.particles_history = []
        self.refuel_events = []
        self.refuel_confidences = []
        
        # For refuel detection
        self.fuel_history = []
        self.speed_history = []
        self.history_window = 20
        self.last_refuel_step = -50
        
        # ESS tracking
        self.ess_history = []
        self.resampled_history = []
        
    def initialize_particles(self, initial_fuel, initial_rate_estimate, 
                            fuel_std=1.0, rate_std=0.0005):
        """
        Initialize particles with Gaussian distribution around initial state
        
        Args:
            initial_fuel: Initial fuel level estimate
            initial_rate_estimate: Initial rate estimate (L/s)
            fuel_std: Standard deviation for fuel initialization
            rate_std: Standard deviation for rate initialization
        """
        self.particles[:, 0] = np.random.normal(initial_fuel, fuel_std, self.num_particles)
        self.particles[:, 1] = np.random.normal(initial_rate_estimate, rate_std, self.num_particles)
        
        # Ensure physical constraints
        self.particles[:, 0] = np.maximum(self.particles[:, 0], 0)  # fuel >= 0
        self.particles[:, 1] = np.maximum(self.particles[:, 1], 0)  # rate >= 0
        
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    # ==================== MOTION MODEL (PREDICTION) ====================
    def motion_model(self, dt):
        """
        MOTION MODEL - Prediction step
        
        This models how the system evolves over time:
        - Fuel decreases based on consumption rate 
        - Rate evolves as random walk
        - Process noise added to model system uncertainty
        
        Physics:
            fuel_{t+1} = fuel_t - rate_t * dt + noise_fuel
            rate_{t+1} = rate_t + noise_rate  (random walk)
        
        Args:
            dt: Time step in seconds
        """
        # Extract current particle states
        fuel = self.particles[:, 0]      # N_particles
        rate = self.particles[:, 1]      # N_particles
        
        # Deterministic part of motion model
        # Fuel consumed in this timestep based on current rate
        fuel_consumed = rate * dt
        new_fuel = fuel - fuel_consumed
        new_fuel = np.maximum(new_fuel, 0)  # Physical constraint
        
        # Rate evolution: Random walk (no speed dependence!)
        # The rate adapts through the measurement update (Bayesian posterior)
        # not through direct speed coupling
        # new_rate = rate.copy()
        new_rate = 0.98 * rate
        
        # Add process noise to model uncertainty in system dynamics
        # Noise on fuel: accounts for measurement errors, sensor lag, etc.
        fuel_noise = np.random.normal(0, self.config.fuel_process_noise, self.num_particles)
        # Noise on rate: allows rate to drift/adapt
        rate_noise = np.random.normal(0, self.config.rate_process_noise, self.num_particles)
        
        self.particles[:, 0] = new_fuel + fuel_noise
        self.particles[:, 1] = new_rate + rate_noise
        
        # Enforce physical constraints
        self.particles[:, 0] = np.maximum(self.particles[:, 0], 0)
        self.particles[:, 1] = np.maximum(self.particles[:, 1], 0)
    
    # ==================== MEASUREMENT MODEL (UPDATE) ====================
    def measurement_model(self, measured_fuel):
        """
        MEASUREMENT MODEL - Correction step
        
        Updates particle weights based on how well each particle's predicted
        fuel matches the actual measured fuel.
        
        Weight Update Rule (Bayesian):
            w_i ∝ p(z_t | x_i) = exp(-0.5 * (measured - predicted)^2 / sigma^2)
        
        Where:
        - z_t: measured fuel level
        - x_i: state of particle i [fuel_i, rate_i]
        - sigma: measurement noise standard deviation
        
        Particles whose predicted fuel is close to measurement get high weight.
        Particles whose prediction is far from measurement get low weight.
        
        Args:
            measured_fuel: Actual fuel measurement from sensor
        """
        # Predicted fuel for each particle
        predicted_fuel = self.particles[:, 0]
        
        # Likelihood: How probable is this measurement given each particle's state?
        # Using Gaussian observation model
        errors = predicted_fuel - measured_fuel
        
        # Log likelihood (for numerical stability)
        log_likelihood = -0.5 * (errors ** 2) / (self.config.measurement_noise ** 2)
        
        likelihood = np.exp(log_likelihood)
        
        # Update weights: multiply old weights by likelihood
        self.weights *= likelihood
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # Fallback: uniform weights
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    # ==================== RESAMPLING ====================
    def systematic_resampling(self):
        """
        RESAMPLING - Particle Propagation step
        
        Replaces low-probability particles with copies of high-probability particles.
        This prevents particle degeneracy (all weight concentrated on few particles).
        
        Algorithm: Systematic Resampling
        1. Compute cumulative distribution of weights
        2. Generate uniform sample positions
        3. Use binary search to select which particles to replicate
        4. Add small jitter (roughening) to replicated particles for diversity
        
        Returns:
            resampled (bool): Whether resampling was triggered
            ess (float): Effective Sample Size before resampling
        """
        # Calculate Effective Sample Size (ESS)
        # ESS = 1 / sum(w_i^2)
        # ESS = N when all weights equal (best case)
        # ESS → 1 when one particle dominates (worst case)
        ess = 1.0 / np.sum(self.weights ** 2)
        
        # Resample if ESS drops below threshold
        ess_threshold = self.config.ess_threshold * self.num_particles
        
        if ess < ess_threshold:
            # === SYSTEMATIC RESAMPLING ===
            # Create cumulative distribution
            cdf = np.cumsum(self.weights)
            
            # Generate uniform sample positions
            u = (np.arange(self.num_particles) + np.random.uniform()) / self.num_particles
            
            # Find which particles to select (binary search)
            indices = np.searchsorted(cdf, u)
            indices = np.clip(indices, 0, self.num_particles - 1)
            
            # Replicate selected particles
            self.particles = self.particles[indices].copy()
            
            # === ROUGHENING (add jitter for diversity) ===
            # Prevents particle collapse to single point
            # Roughening amount depends on state spread
            fuel_percentiles = np.percentile(self.particles[:, 0], [5, 95])
            rate_percentiles = np.percentile(self.particles[:, 1], [5, 95])
            
            fuel_span = fuel_percentiles[1] - fuel_percentiles[0]
            rate_span = rate_percentiles[1] - rate_percentiles[0]
            
            # Roughening proportional to state space span and inversely to N^(1/d)
            # d=2 (state dimension)
            roughening_factor = 0.01 / (self.num_particles ** (1/2))
            
            fuel_roughen_sigma = max(0.1, roughening_factor * fuel_span)
            # fuel_roughen_sigma = roughening_factor * fuel_span * 0.3
            rate_roughen_sigma = max(0.00001, roughening_factor * rate_span)
            
            self.particles[:, 0] += np.random.normal(0, fuel_roughen_sigma, self.num_particles)
            self.particles[:, 1] += np.random.normal(0, rate_roughen_sigma, self.num_particles)
            
            # Re-enforce constraints
            self.particles[:, 0] = np.maximum(self.particles[:, 0], 0)
            self.particles[:, 1] = np.maximum(self.particles[:, 1], 0)
            
            # Reset weights to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
            
            return True, ess
        
        return False, ess
    
    # ==================== REFUEL DETECTION ====================
    def detect_refuel(self, prev_fuel, curr_fuel, curr_speed, step_num):
        """
        Detect refueling events
        
        Heuristics:
        1. Large positive jump in fuel (>5L)
        2. Robot is stopped (speed < 0.15 m/s)
        3. Recent speed history shows stopped state
        4. Fuel was stable or decreasing before jump
        """
        self.fuel_history.append(curr_fuel)
        self.speed_history.append(curr_speed)
        
        if len(self.fuel_history) > self.history_window:
            self.fuel_history.pop(0)
            self.speed_history.pop(0)
        
        if len(self.fuel_history) < 5:
            return False, 0.0
        
        fuel_jump = curr_fuel - prev_fuel
        
        # 1. Minimum jump threshold
        if fuel_jump < 5.0:
            return False, 0.0
        
        # 2. Robot should be stopped
        if abs(curr_speed) > 0.15:
            return False, 0.0
        
        # 3. Check recent speed history
        recent_speeds = self.speed_history[-5:]
        stopped_count = sum(1 for s in recent_speeds if abs(s) < 0.15)
        if stopped_count < 3:
            return False, 0.0
        
        # 4. Fuel was stable before
        prev_changes = [self.fuel_history[i] - self.fuel_history[i-1]
                       for i in range(len(self.fuel_history)-4, len(self.fuel_history)-1)]
        if any(change > 3.0 for change in prev_changes):
            return False, 0.0
        
        # Calculate confidence
        jump_confidence = min(1.0, fuel_jump / 15.0)
        stop_confidence = stopped_count / 5.0
        confidence = 0.6 * jump_confidence + 0.4 * stop_confidence
        
        self.last_refuel_step = step_num
        # print(f"✓ Refuel detected at step {step_num}: "
        #       f"{prev_fuel:.2f}L → {curr_fuel:.2f}L "
        #       f"(jump: {fuel_jump:.2f}L, confidence: {confidence:.2f})")
        
        return True, confidence
    
    # ==================== MAIN FILTER STEP ====================
    def step(self, prev_fuel, curr_fuel, curr_speed, curr_timestamp, dt, step_num=0):
        """
        Execute one complete filter iteration: PREDICT -> UPDATE -> RESAMPLE
        
        Cycle:
        1. MOTION MODEL: Predict how fuel and rate evolve
        2. MEASUREMENT MODEL: Weight particles by observation likelihood
        3. RESAMPLING: Replace low-weight particles with high-weight ones
        
        Args:
            prev_fuel: Fuel measurement from previous step
            curr_fuel: Current fuel measurement
            curr_speed: Current robot speed
            curr_timestamp: Timestamp
            dt: Time step in seconds
            step_num: Step number
            
        Returns:
            fuel_est: Estimated fuel (weighted mean)
            rate_est: Estimated consumption rate (weighted mean)
            fuel_std: Standard deviation of fuel estimate
            is_refuel: Whether refuel detected
            confidence: Refuel detection confidence
        """
        # Detect refuel
        is_refuel, confidence = self.detect_refuel(prev_fuel, curr_fuel, curr_speed, step_num)
        
        # On refuel: reset particle distribution around measured fuel
        if is_refuel:
            reset_ratio = 0.5 + 0.3 * confidence
            n_reset = int(self.num_particles * reset_ratio)
            reset_indices = np.random.choice(self.num_particles, n_reset, replace=False)
            
            # Reset fuel particles around measured value
            self.particles[reset_indices, 0] = np.random.normal(
                curr_fuel, 0.5, n_reset
            )
            # Reset rate particles to uncertainty distribution
            self.particles[reset_indices, 1] = np.random.normal(
                np.mean(self.particles[:, 1]), 0.0001, n_reset
            )
            
            self.particles[:, 0] = np.maximum(self.particles[:, 0], 0)
            self.particles[:, 1] = np.maximum(self.particles[:, 1], 0)
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # ===== STEP 1: MOTION MODEL (PREDICT) =====
        self.motion_model(dt)
        
        # ===== STEP 2: MEASUREMENT MODEL (UPDATE) =====
        self.measurement_model(curr_fuel)
        
        # ===== STEP 3: RESAMPLING =====
        resampled, ess = self.systematic_resampling()
        
        # Compute estimates
        fuel_est = np.average(self.particles[:, 0], weights=self.weights)
        rate_est = np.average(self.particles[:, 1], weights=self.weights)
        
        # Compute variance (uncertainty)
        fuel_var = np.average((self.particles[:, 0] - fuel_est)**2, weights=self.weights)
        rate_var = np.average((self.particles[:, 1] - rate_est)**2, weights=self.weights)
        fuel_std = np.sqrt(fuel_var)
        rate_std = np.sqrt(rate_var)
        
        # Store history
        self.estimates_mean.append([fuel_est, rate_est])
        self.estimates_cov.append([[fuel_var, 0], [0, rate_var]])
        self.ess_history.append(ess)
        self.resampled_history.append(resampled)
        
        # Store particle percentiles for visualization
        fuel_p5 = np.percentile(self.particles[:, 0], 5)
        fuel_p25 = np.percentile(self.particles[:, 0], 25)
        fuel_p50 = np.percentile(self.particles[:, 0], 50)
        fuel_p75 = np.percentile(self.particles[:, 0], 75)
        fuel_p95 = np.percentile(self.particles[:, 0], 95)
        
        rate_p5 = np.percentile(self.particles[:, 1], 5)
        rate_p25 = np.percentile(self.particles[:, 1], 25)
        rate_p75 = np.percentile(self.particles[:, 1], 75)
        rate_p95 = np.percentile(self.particles[:, 1], 95)
        
        self.particles_history.append({
            'fuel_p5': fuel_p5, 'fuel_p25': fuel_p25, 'fuel_p50': fuel_p50,
            'fuel_p75': fuel_p75, 'fuel_p95': fuel_p95,
            'rate_p5': rate_p5,
            'rate_p25': rate_p25,
            'rate_p75': rate_p75,
            'rate_p95': rate_p95,
            'fuel_std': fuel_std, 'rate_std': rate_std,
            'is_refuel': is_refuel, 'confidence': confidence,
            'ess': ess, 'resampled': resampled,
            'particles': self.particles.copy()
        })
        
        if is_refuel:
            self.refuel_events.append(step_num)
            self.refuel_confidences.append(confidence)
        
        return fuel_est, rate_est, fuel_std, is_refuel, confidence


def load_data(csv_path='fuel_cmd.csv', nrows=None):
    """Load and preprocess fuel data"""
    fuel_col = 'ros_main__generator_controller__hatz_info__fuel_level__value'
    speed_col = 'ros_main__inverse_kinematics__cmd__data__speed'
    
    df = pd.read_csv(csv_path, nrows=nrows, parse_dates=['datetime'], dayfirst=False)
    
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    df_clean = pd.DataFrame({
        'fuel_level': df[fuel_col].fillna(0),
        'speed': df[speed_col].fillna(0),
        'timestamp': df['timestamp'],
        'datetime': df['datetime']
    })
    
    df_clean = df_clean[df_clean['fuel_level'] > 0].reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_clean):,} measurements")
    print(f"  Fuel: {df_clean['fuel_level'].min():.2f} - {df_clean['fuel_level'].max():.2f} L")
    print(f"  Speed: {df_clean['speed'].min():.2f} - {df_clean['speed'].max():.2f} m/s")
    
    return df_clean


def run_filter(df, config: ParticleFilterConfig):
    """Run the 2D particle filter"""
    
    pf = ParticleFilter2D(config)
    
    # Initialize around first measurement with reasonable initial rate
    initial_fuel = df.iloc[0]['fuel_level']
    initial_rate = 0.005  # 5 mL/s estimate
    pf.initialize_particles(initial_fuel, initial_rate)
    
    fuel_estimates = []
    rate_estimates = []
    fuel_stds = []
    refuel_count = 0
    
    print("\nProcessing measurements with 2D Particle Filter...")
    
    for i in range(1, len(df)):
        # iterating through time stamps
        curr_fuel = df.iloc[i]['fuel_level']
        curr_speed = df.iloc[i]['speed']
        prev_fuel = df.iloc[i-1]['fuel_level']
        
        # Compute dt
        dt = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp'])
        if dt <= 0:
            dt = 1.0
        
        # Run filter step
        fuel_est, rate_est, fuel_std, is_refuel, confidence = pf.step(
            prev_fuel, curr_fuel, curr_speed, df.iloc[i]['datetime'], dt, step_num=i
        )
        fuel_estimates.append(fuel_est)
        rate_estimates.append(rate_est)
        fuel_stds.append(fuel_std)
        
        if is_refuel:
            refuel_count += 1
        
        # Progress update
        if (i + 1) % (len(df) // 5) == 0 or i == 1:
            pct = 100 * (i + 1) / len(df)
            error = abs(fuel_est - curr_fuel)
            print(f"{pct:3.0f}% - Est: {fuel_est:.2f}L | "
                  f"Sensor: {curr_fuel:.2f}L | Error: {error:.2f}L | "
                  f"Rate: {rate_est*3600:.4f}L/h | "
                  f"Refuels: {refuel_count}")
    
    # Calculate metrics
    error = df['fuel_level'].values[1:] - np.array(fuel_estimates)
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    
    print(f"\n{'='*70}")
    print("2D PARTICLE FILTER RESULTS:")
    print(f"  RMSE: {rmse:.4f}L")
    print(f"  MAE:  {mae:.4f}L")
    print(f"  Mean Rate: {np.mean(rate_estimates)*3600:.4f} L/hour")
    print(f"  Std Rate: {np.std(rate_estimates)*3600:.4f} L/hour")
    print(f"  Refueling events: {refuel_count}")
    print(f"  Resampling events: {sum(pf.resampled_history)}")
    print(f"{'='*70}\n")
    
    return {
        'pf': pf,
        'df': df[1:].reset_index(drop=True),
        'fuel_estimates': np.array(fuel_estimates),
        'rate_estimates': np.array(rate_estimates),
        'fuel_stds': np.array(fuel_stds),
        'rmse': rmse,
        'mae': mae,
        'refuel_count': refuel_count
    }


def plot_results(results, output_file='particle_filter_2d.html'):
    """Create comprehensive visualization"""
    pf = results['pf']
    df = results['df']
    fuel_est = results['fuel_estimates']
    rate_est = results['rate_estimates']
    fuel_std = results['fuel_stds']
    rmse = results['rmse']
    mae = results['mae']
    refuel_count = results['refuel_count']
    
    time = pd.to_datetime(
        df['datetime'] if 'datetime' in df.columns
        else pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    )
    error = df['fuel_level'].values - fuel_est
    
    # Extract particle percentiles
    particles_hist = pf.particles_history
    # fuel_p5 = np.array([p['fuel_p5'] for p in particles_hist])
    # fuel_p25 = np.array([p['fuel_p25'] for p in particles_hist])
    # fuel_p75 = np.array([p['fuel_p75'] for p in particles_hist])
    # fuel_p95 = np.array([p['fuel_p95'] for p in particles_hist])
    # rate_p5 = np.array([p['rate_p5'] for p in particles_hist])
    # rate_p25 = np.array([p['rate_p25'] for p in particles_hist])
    # rate_p75 = np.array([p['rate_p75'] for p in particles_hist])
    # rate_p95 = np.array([p['rate_p95'] for p in particles_hist])
    # fuel_std_arr = np.array([p['fuel_std'] for p in particles_hist])
    # ess_arr = np.array([p['ess'] for p in particles_hist])
    # resampled_arr = np.array([p['resampled'] for p in particles_hist])

    # num_particles = pf.num_particles

    # # # Build trajectories: shape (num_particles, T)
    # fuel_particles_traj = np.zeros((num_particles, len(particles_hist)))
    # rate_particles_traj = np.zeros((num_particles, len(particles_hist)))

    # for t, ph in enumerate(particles_hist):
    #     # ph['particles'] has shape (num_particles, 2)
    #     fuel_particles_traj[:, t] = ph['particles'][:, 0]
    #     rate_particles_traj[:, t] = ph['particles'][:, 1] * 3600.0  # convert to L/h
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"Fuel Estimation (Refuels: {refuel_count})",
            "Consumption Rate (L/hour)",
            "Estimation Error"
        ),
        row_heights=[0.25, 0.20, 0.25],
        vertical_spacing=0.08
    )
    
    # Row 1: Fuel estimation with particle bands
    fig.add_trace(
        go.Scatter(x=time, y=df['fuel_level'], mode='lines',
                   name='Sensor', line=dict(color='gray', width=1.5)),
        row=1, col=1
    )
    
    # fig.add_trace(
    #     go.Scatter(x=time, y=fuel_p95, fill=None, mode='lines',
    #                line_color='rgba(0,0,0,0)', showlegend=False),
    #     row=1, col=1
    # )
    # fig.add_trace(
    #     go.Scatter(x=time, y=fuel_p5, fill='tonexty', mode='lines',
    #                line_color='rgba(0,0,0,0)', fillcolor='rgba(100,149,237,0.2)',
    #                name='5-95% Particle Range'),
    #     row=1, col=1
    # )
    
    # fig.add_trace(
    #     go.Scatter(x=time, y=fuel_p75, fill=None, mode='lines',
    #                line_color='rgba(0,0,0,0)', showlegend=False),
    #     row=1, col=1
    # )
    # fig.add_trace(
    #     go.Scatter(x=time, y=fuel_p25, fill='tonexty', mode='lines',
    #                line_color='rgba(0,0,0,0)', fillcolor='rgba(30,144,255,0.3)',
    #                name='25-75% IQR'),
    #     row=1, col=1
    # )

    # for pid in range(num_particles):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=time,
    #             y=fuel_particles_traj[pid, :],
    #             mode='lines+markers',
    #             name=f'Particle {pid} Fuel',
    #             line=dict(width=1),
    #             marker=dict(size=5),
    #             opacity=0.7,
    #             showlegend=(pid == 0),  # only one legend entry to avoid clutter
    #             legendgroup="particles_fuel"
    #         ),
    #         row=1, col=1
    #     )
    
    fig.add_trace(
        go.Scatter(x=time, y=fuel_est, mode='lines',
                   name='PF Estimate', line=dict(color='orange', width=2.5)),
        row=1, col=1
    )
    
    # Mark refuels
    if pf.refuel_events:
        refuel_indices = [idx-1 for idx in pf.refuel_events if idx-1 < len(time)]
        refuel_times = [time[idx] for idx in refuel_indices]
        refuel_values = [df['fuel_level'].iloc[idx] for idx in refuel_indices]
        
        fig.add_trace(
            go.Scatter(x=refuel_times, y=refuel_values, mode='markers',
                       name='Refuel Events',
                       marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )
    
    # Row 2: Uncertainty (standard deviation)
    # fig.add_trace(
    #     go.Scatter(x=time, y=fuel_std_arr, mode='lines',
    #                name='Fuel Estimate Std Dev',
    #                line=dict(color='purple', width=2), fill='tozeroy',
    #                fillcolor='rgba(128,0,128,0.2)'),
    #     row=2, col=1
    # )
    
    # Row 3: Consumption rate
    fig.add_trace(
        go.Scatter(x=time, y=rate_est*3600, mode='lines',
                   name='Estimated Rate',
                   line=dict(color='green', width=2)),
        row=2, col=1
    )

    # fig.add_trace(
    #     go.Scatter(x=time, y=rate_p95*3600, fill=None, mode='lines',
    #                line_color='rgba(0,0,0,0)', showlegend=False),
    #     row=2, col=1
    # )

    # fig.add_trace(
    #     go.Scatter(x=time, y=rate_p5*3600, fill='tonexty', mode='lines',
    #                line_color='rgba(0,0,0,0)', fillcolor='rgba(100,149,237,0.2)',
    #                name='5-95% Rate Particle Range'),
    #     row=2, col=1
    # )

    # fig.add_trace(
    #     go.Scatter(x=time, y=rate_p75*3600, fill=None, mode='lines',
    #                line_color='rgba(0,0,0,0)', showlegend=False),
    #     row=2, col=1
    # )
    # fig.add_trace(
    #     go.Scatter(x=time, y=rate_p25*3600, fill='tonexty', mode='lines',
    #                line_color='rgba(0,0,0,0)', fillcolor='rgba(30,144,255,0.3)',
    #                name='25-75% Rate Particle IQR'),
    #     row=2, col=1
    # )

    # for pid in range(num_particles):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=time,
    #             y=rate_particles_traj[pid, :],
    #             mode='lines+markers',
    #             name=f'Particle {pid} Rate',
    #             line=dict(width=1),
    #             marker=dict(size=5),
    #             opacity=0.7,
    #             showlegend=(pid == 0),  # only one legend entry
    #             legendgroup="particles_rate"
    #         ),
    #         row=2, col=1
    #     )
    
    # Row 4: ESS
    # fig.add_trace(
    #     go.Scatter(x=time, y=ess_arr, mode='lines',
    #                name='ESS',
    #                line=dict(color='blue', width=1.5), fill='tozeroy',
    #                fillcolor='rgba(0,0,255,0.1)'),
    #     row=4, col=1
    # )
    # ess_threshold = pf.config.ess_threshold * pf.num_particles
    # fig.add_hline(y=ess_threshold, line_dash="dash", line_color="red",
    #               row=4, col=1, opacity=0.5, annotation_text="Resample Threshold")
    
    # Row 5: Error
    fig.add_trace(
        go.Scatter(x=time, y=error, mode='lines',
                   name='Estimation Error',
                   line=dict(color='red', width=1.5), fill='tozeroy',
                   fillcolor='rgba(255,0,0,0.15)'),
        row=3, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1, opacity=0.3)
    
    # Update layout
    fig.update_layout(
        title=f"2D Particle Filter [Fuel, Rate] | RMSE: {rmse:.3f}L | MAE: {mae:.3f}L",
        height=1200,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            x=1.02, y=1.0,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
    )
    
    fig.update_yaxes(title_text="Fuel (L)", row=1, col=1)
    # fig.update_yaxes(title_text="Std Dev (L)", row=2, col=1)
    fig.update_yaxes(title_text="Rate (L/h)", row=2, col=1)
    # fig.update_yaxes(title_text="ESS", row=4, col=1)
    fig.update_yaxes(title_text="Error (L)", row=3, col=1)
    
    for r in range(1, 6):
        fig.update_xaxes(title_text="Time", row=r, col=1)
    
    fig.write_html(output_file)
    print(f"✓ Visualization saved: {output_file}")


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("2D PARTICLE FILTER FOR FUEL LEVEL ESTIMATION")
    print("State: [fuel (L), consumption_rate (L/s)]")
    print("="*70)
    
    # Configuration
    # config = ParticleFilterConfig(
    #     num_particles=300,
    #     fuel_process_noise=1.5,      
    #     rate_process_noise=1e-4,      
    #     measurement_noise=5.0,        
    #     ess_threshold=0.5
    # )

    config = ParticleFilterConfig(
        num_particles=300,
        fuel_process_noise=0.15,      
        rate_process_noise=5e-4,      
        measurement_noise=1.5,        
        ess_threshold=0.5
    )
    
    # Load data
    df = load_data('fuel_cmd.csv')
    day_start = pd.Timestamp('2025-08-15 00:00:00')
    day_end   = day_start + pd.Timedelta(days=10)

    df_day = df[(df['datetime'] >= day_start) & (df['datetime'] < day_end)].reset_index(drop=True)
    
    # Run filter
    # results = run_filter(df, config)
    results = run_filter(df_day, config)
    
    # Visualize
    plot_results(results, 'particle_filter.html')
    
    print("✓ Complete!\n")


if __name__ == '__main__':
    main()