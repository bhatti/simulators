"""
Capacity Planning Simulator
Combines queueing theory (Little's Law, M/M/c models) with discrete event simulation
to help predict and visualize system capacity, queue delays, and utilization.

Key Concepts:
- Little's Law: L = λW (avg queue length = arrival rate × avg wait time)
- M/M/c Queue: Multiple servers, Poisson arrivals, exponential service
- Utilization: ρ = λ / (c × μ) where c = servers, μ = service rate
- Erlang C: Probability of queueing in M/M/c system

Based on research and practical capacity planning for distributed systems.
"""

import streamlit as st
import simpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from scipy import special

# ============================================================================
# QUEUEING THEORY CALCULATIONS
# ============================================================================

class QueueingTheory:
    """
    Theoretical calculations for M/M/c queues.
    
    M/M/c: Markovian arrivals (Poisson), Markovian service (Exponential), c servers
    
    Parameters:
    - λ (lambda): Arrival rate (requests per unit time)
    - μ (mu): Service rate per server (requests per unit time)
    - c: Number of servers
    """
    
    @staticmethod
    def utilization(arrival_rate: float, service_rate: float, num_servers: int) -> float:
        """
        Calculate server utilization (ρ).
        ρ = λ / (c × μ)
        
        Must be < 1 for stable queue, otherwise queue grows unbounded.
        """
        if num_servers <= 0 or service_rate <= 0:
            return float('inf')
        return arrival_rate / (num_servers * service_rate)
    
    @staticmethod
    def erlang_c(arrival_rate: float, service_rate: float, num_servers: int) -> float:
        """
        Erlang C formula: Probability that an arriving customer has to wait.
        
        This is the key formula for capacity planning - tells you what fraction
        of requests will experience queueing delay.
        """
        c = num_servers
        rho = QueueingTheory.utilization(arrival_rate, service_rate, c)
        
        if rho >= 1:
            return 1.0  # System unstable, everyone waits
        
        # Calculate (c*rho)^c / c!
        a = arrival_rate / service_rate  # Offered load
        
        # Sum for denominator
        sum_term = sum((a ** k) / math.factorial(k) for k in range(c))
        
        # Final term
        last_term = (a ** c) / (math.factorial(c) * (1 - rho))
        
        # Erlang C probability
        erlang_c = last_term / (sum_term + last_term)
        
        return erlang_c
    
    @staticmethod
    def avg_queue_length(arrival_rate: float, service_rate: float, num_servers: int) -> float:
        """
        Average number of requests waiting in queue (Lq).
        
        Lq = Erlang_C × ρ / (1 - ρ)
        """
        rho = QueueingTheory.utilization(arrival_rate, service_rate, num_servers)
        
        if rho >= 1:
            return float('inf')
        
        erlang_c = QueueingTheory.erlang_c(arrival_rate, service_rate, num_servers)
        
        return erlang_c * rho / (1 - rho)
    
    @staticmethod
    def avg_wait_time(arrival_rate: float, service_rate: float, num_servers: int) -> float:
        """
        Average time spent waiting in queue (Wq).
        
        Using Little's Law: Wq = Lq / λ
        """
        if arrival_rate <= 0:
            return 0
        
        lq = QueueingTheory.avg_queue_length(arrival_rate, service_rate, num_servers)
        
        if lq == float('inf'):
            return float('inf')
        
        return lq / arrival_rate
    
    @staticmethod
    def avg_system_time(arrival_rate: float, service_rate: float, num_servers: int) -> float:
        """
        Average total time in system (W = Wq + 1/μ).
        
        Includes both waiting time and service time.
        """
        wq = QueueingTheory.avg_wait_time(arrival_rate, service_rate, num_servers)
        
        if wq == float('inf'):
            return float('inf')
        
        return wq + (1 / service_rate)
    
    @staticmethod
    def avg_requests_in_system(arrival_rate: float, service_rate: float, num_servers: int) -> float:
        """
        Average number of requests in system (L = Lq + λ/μ).
        
        Little's Law: L = λW
        """
        lq = QueueingTheory.avg_queue_length(arrival_rate, service_rate, num_servers)
        
        if lq == float('inf'):
            return float('inf')
        
        return lq + (arrival_rate / service_rate)
    
    @staticmethod
    def percentile_wait_time(arrival_rate: float, service_rate: float, 
                             num_servers: int, percentile: float = 0.95) -> float:
        """
        Estimate percentile wait time using exponential approximation.
        
        For M/M/c queues, the conditional wait time (given you wait) is 
        approximately exponential with rate c*μ - λ.
        """
        rho = QueueingTheory.utilization(arrival_rate, service_rate, num_servers)
        
        if rho >= 1:
            return float('inf')
        
        erlang_c = QueueingTheory.erlang_c(arrival_rate, service_rate, num_servers)
        
        if erlang_c <= 0:
            return 0
        
        # Rate of conditional wait time distribution
        rate = num_servers * service_rate - arrival_rate
        
        if rate <= 0:
            return float('inf')
        
        # P(W > t) = Erlang_C × exp(-rate × t)
        # Solve for t at given percentile
        # percentile = 1 - Erlang_C × exp(-rate × t)
        # exp(-rate × t) = (1 - percentile) / Erlang_C
        
        if (1 - percentile) / erlang_c >= 1:
            return 0  # Percentile is below the probability of waiting
        
        if (1 - percentile) / erlang_c <= 0:
            return float('inf')
        
        t = -math.log((1 - percentile) / erlang_c) / rate
        
        return max(0, t)
    
    @staticmethod
    def required_servers(arrival_rate: float, service_rate: float, 
                        target_utilization: float = 0.7) -> int:
        """
        Calculate required number of servers for target utilization.
        
        c = λ / (ρ_target × μ)
        """
        if service_rate <= 0 or target_utilization <= 0:
            return 1
        
        required = arrival_rate / (target_utilization * service_rate)
        return max(1, math.ceil(required))
    
    @staticmethod
    def max_arrival_rate(service_rate: float, num_servers: int, 
                        target_utilization: float = 0.8) -> float:
        """
        Calculate maximum sustainable arrival rate for given capacity.
        
        λ_max = ρ_target × c × μ
        """
        return target_utilization * num_servers * service_rate


# ============================================================================
# SIMULATION COMPONENTS
# ============================================================================

class LoadPattern(Enum):
    """Different load patterns for simulation"""
    CONSTANT = "Constant"
    RAMP_UP = "Ramp Up"
    RAMP_UP_DOWN = "Ramp Up/Down"
    SPIKE = "Traffic Spike"
    DIURNAL = "Diurnal (Day/Night)"
    BURSTY = "Bursty"

@dataclass
class SimulationConfig:
    """Configuration for capacity simulation"""
    # Arrival parameters
    base_arrival_rate: float = 50.0  # requests per minute
    
    # Service parameters
    avg_service_time: float = 0.5  # seconds
    service_time_variance: float = 0.2  # coefficient of variation
    
    # Dependency parameters
    num_dependencies: int = 2
    avg_dependency_latency: float = 0.1  # seconds per dependency
    dependency_failure_rate: float = 0.01  # probability of dependency failure
    
    # Server parameters
    num_servers: int = 5
    max_queue_size: Optional[int] = None  # None = unlimited
    
    # Retry parameters
    enable_retries: bool = True
    max_retries: int = 3
    retry_backoff_base: float = 0.1
    retry_backoff_enabled: bool = True
    
    # Circuit breaker
    enable_circuit_breaker: bool = False
    circuit_breaker_threshold: float = 0.5  # failure rate to open
    circuit_breaker_recovery_time: float = 30.0  # seconds
    
    # Auto-scaling
    enable_auto_scaling: bool = False
    scale_up_threshold: float = 0.8  # utilization to scale up
    scale_down_threshold: float = 0.3  # utilization to scale down
    scale_cooldown: float = 60.0  # seconds between scaling actions
    min_servers: int = 2
    max_servers: int = 20
    
    # Load pattern
    load_pattern: LoadPattern = LoadPattern.CONSTANT
    spike_multiplier: float = 3.0
    spike_start: float = 30.0
    spike_duration: float = 20.0
    ramp_peak_time: float = 60.0
    
    # Simulation
    duration_minutes: float = 5.0
    random_seed: int = 42

@dataclass
class SimulationMetrics:
    """Collected metrics from simulation"""
    timestamps: List[float] = field(default_factory=list)
    queue_lengths: List[int] = field(default_factory=list)
    utilizations: List[float] = field(default_factory=list)
    active_servers: List[int] = field(default_factory=list)
    
    wait_times: List[float] = field(default_factory=list)
    service_times: List[float] = field(default_factory=list)
    system_times: List[float] = field(default_factory=list)
    
    arrivals: int = 0
    completions: int = 0
    rejections: int = 0  # Due to full queue
    timeouts: int = 0
    retries: int = 0
    dependency_failures: int = 0
    circuit_breaker_rejections: int = 0
    
    arrival_timestamps: List[float] = field(default_factory=list)
    arrival_rates: List[float] = field(default_factory=list)


class CapacitySimulator:
    """
    Discrete event simulation for capacity planning.
    Models servers, queues, dependencies, retries, and auto-scaling.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.env = None
        self.servers = None
        self.metrics = None
        self.circuit_breaker_open = False
        self.circuit_breaker_open_time = 0
        self.last_scale_time = 0
        self.current_failure_rate = 0
        self.recent_failures = []
        self.recent_successes = []
        
    def get_service_time(self) -> float:
        """Generate service time with variance"""
        # Use gamma distribution for more realistic service times
        mean = self.config.avg_service_time
        cv = self.config.service_time_variance  # coefficient of variation
        
        if cv <= 0:
            return mean
        
        # Gamma parameters
        shape = 1 / (cv ** 2)
        scale = mean / shape
        
        return random.gammavariate(shape, scale)
    
    def get_dependency_time(self) -> Tuple[float, bool]:
        """
        Generate dependency latency and check for failures.
        Returns (total_time, success)
        """
        total_time = 0
        
        for _ in range(self.config.num_dependencies):
            if random.random() < self.config.dependency_failure_rate:
                return (total_time, False)
            
            # Exponential distribution for dependency latency
            dep_time = random.expovariate(1.0 / self.config.avg_dependency_latency)
            total_time += dep_time
        
        return (total_time, True)
    
    def get_arrival_rate(self, current_time: float) -> float:
        """Get arrival rate based on load pattern and current time"""
        base_rate = self.config.base_arrival_rate / 60.0  # Convert to per second
        
        if self.config.load_pattern == LoadPattern.CONSTANT:
            return base_rate
        
        elif self.config.load_pattern == LoadPattern.RAMP_UP:
            # Linear ramp from 0 to base over ramp_peak_time
            progress = min(1.0, current_time / self.config.ramp_peak_time)
            return base_rate * progress
        
        elif self.config.load_pattern == LoadPattern.RAMP_UP_DOWN:
            # Ramp up to peak, then back down
            peak_time = self.config.ramp_peak_time
            if current_time <= peak_time:
                progress = current_time / peak_time
            else:
                progress = max(0, 1 - (current_time - peak_time) / peak_time)
            return base_rate * (0.2 + 0.8 * progress)  # Min 20% of base
        
        elif self.config.load_pattern == LoadPattern.SPIKE:
            # Sudden spike for duration
            spike_start = self.config.spike_start
            spike_end = spike_start + self.config.spike_duration
            
            if spike_start <= current_time < spike_end:
                return base_rate * self.config.spike_multiplier
            return base_rate
        
        elif self.config.load_pattern == LoadPattern.DIURNAL:
            # Sinusoidal pattern (day/night cycle)
            # One full cycle over simulation duration
            duration = self.config.duration_minutes * 60
            phase = 2 * math.pi * current_time / duration
            multiplier = 0.5 + 0.5 * math.sin(phase)  # 0 to 1
            return base_rate * (0.3 + 0.7 * multiplier)  # 30% to 100%
        
        elif self.config.load_pattern == LoadPattern.BURSTY:
            # Random bursts
            # Use a modulated Poisson process
            burst_probability = 0.1
            if random.random() < burst_probability:
                return base_rate * self.config.spike_multiplier
            return base_rate
        
        return base_rate
    
    def check_circuit_breaker(self, current_time: float) -> bool:
        """
        Check if circuit breaker should allow request.
        Returns True if request should be rejected.
        """
        if not self.config.enable_circuit_breaker:
            return False
        
        # Check if we should close the circuit
        if self.circuit_breaker_open:
            if current_time - self.circuit_breaker_open_time > self.config.circuit_breaker_recovery_time:
                self.circuit_breaker_open = False
            else:
                return True
        
        # Update failure rate (rolling window)
        window = 10.0  # 10 second window
        self.recent_failures = [t for t in self.recent_failures if current_time - t < window]
        self.recent_successes = [t for t in self.recent_successes if current_time - t < window]
        
        total = len(self.recent_failures) + len(self.recent_successes)
        if total > 10:  # Need minimum samples
            failure_rate = len(self.recent_failures) / total
            if failure_rate > self.config.circuit_breaker_threshold:
                self.circuit_breaker_open = True
                self.circuit_breaker_open_time = current_time
                return True
        
        return False
    
    def check_auto_scaling(self, current_time: float) -> int:
        """
        Check if auto-scaling should adjust server count.
        Returns new server count.
        """
        if not self.config.enable_auto_scaling:
            return self.servers.capacity
        
        # Cooldown check
        if current_time - self.last_scale_time < self.config.scale_cooldown:
            return self.servers.capacity
        
        # Calculate current utilization
        current_capacity = self.servers.capacity
        utilization = self.servers.count / current_capacity if current_capacity > 0 else 1.0
        
        new_capacity = current_capacity
        
        if utilization > self.config.scale_up_threshold:
            # Scale up
            new_capacity = min(self.config.max_servers, current_capacity + 1)
        elif utilization < self.config.scale_down_threshold:
            # Scale down
            new_capacity = max(self.config.min_servers, current_capacity - 1)
        
        if new_capacity != current_capacity:
            self.last_scale_time = current_time
            # Note: simpy Resource capacity can't be changed dynamically
            # This is a simplification - in reality we'd need more complex handling
        
        return new_capacity
    
    def request_process(self, env, request_id: int):
        """Process a single request through the system"""
        arrival_time = env.now
        self.metrics.arrivals += 1
        
        # Check circuit breaker
        if self.check_circuit_breaker(arrival_time):
            self.metrics.circuit_breaker_rejections += 1
            return
        
        # Check queue capacity
        if self.config.max_queue_size is not None:
            if len(self.servers.queue) >= self.config.max_queue_size:
                self.metrics.rejections += 1
                return
        
        retries_left = self.config.max_retries if self.config.enable_retries else 0
        attempt = 0
        
        while True:
            attempt += 1
            
            with self.servers.request() as req:
                yield req
                
                # Record wait time
                wait_time = env.now - arrival_time
                self.metrics.wait_times.append(wait_time)
                
                # Service time
                service_start = env.now
                service_time = self.get_service_time()
                yield env.timeout(service_time)
                
                # Dependency calls
                dep_time, dep_success = self.get_dependency_time()
                yield env.timeout(dep_time)
                
                if not dep_success:
                    self.metrics.dependency_failures += 1
                    self.recent_failures.append(env.now)
                    
                    if retries_left > 0:
                        retries_left -= 1
                        self.metrics.retries += 1
                        
                        if self.config.retry_backoff_enabled:
                            backoff = self.config.retry_backoff_base * (2 ** attempt) * random.uniform(0.5, 1.5)
                            yield env.timeout(backoff)
                        
                        arrival_time = env.now  # Reset for retry
                        continue
                    else:
                        self.metrics.timeouts += 1
                        return
                
                # Success!
                self.metrics.completions += 1
                self.recent_successes.append(env.now)
                
                total_service = env.now - service_start
                self.metrics.service_times.append(total_service)
                
                system_time = env.now - arrival_time
                self.metrics.system_times.append(system_time)
                
                return
    
    def arrival_process(self, env):
        """Generate arrivals according to load pattern"""
        request_id = 0
        
        while True:
            # Get current arrival rate
            current_rate = self.get_arrival_rate(env.now)
            
            if current_rate > 0:
                # Inter-arrival time (exponential for Poisson process)
                inter_arrival = random.expovariate(current_rate)
                yield env.timeout(inter_arrival)
                
                # Record arrival rate
                self.metrics.arrival_timestamps.append(env.now)
                self.metrics.arrival_rates.append(current_rate * 60)  # Convert to per minute
                
                # Spawn request
                request_id += 1
                env.process(self.request_process(env, request_id))
            else:
                yield env.timeout(1.0)  # Wait if no arrivals
    
    def monitor_process(self, env, interval: float = 1.0):
        """Periodically record metrics"""
        while True:
            self.metrics.timestamps.append(env.now)
            self.metrics.queue_lengths.append(len(self.servers.queue))
            
            capacity = max(1, self.servers.capacity)
            utilization = (self.servers.count / capacity) * 100
            self.metrics.utilizations.append(utilization)
            self.metrics.active_servers.append(self.servers.count)
            
            # Check auto-scaling
            if self.config.enable_auto_scaling:
                new_capacity = self.check_auto_scaling(env.now)
                # Note: Can't actually change capacity in simpy easily
                # This is recorded for visualization
            
            yield env.timeout(interval)
    
    def run(self) -> SimulationMetrics:
        """Run the simulation and return metrics"""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        self.env = simpy.Environment()
        self.servers = simpy.Resource(self.env, capacity=self.config.num_servers)
        self.metrics = SimulationMetrics()
        
        # Reset state
        self.circuit_breaker_open = False
        self.last_scale_time = 0
        self.recent_failures = []
        self.recent_successes = []
        
        # Start processes
        self.env.process(self.arrival_process(self.env))
        self.env.process(self.monitor_process(self.env))
        
        # Run simulation
        duration_seconds = self.config.duration_minutes * 60
        self.env.run(until=duration_seconds)
        
        return self.metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_theory_vs_simulation_plot(theory_results: dict, sim_metrics: SimulationMetrics, 
                                     config: SimulationConfig) -> go.Figure:
    """Create comparison plot of theory vs simulation"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Queue Length Over Time',
            'Server Utilization Over Time',
            'Wait Time Distribution',
            'System Time Distribution'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )
    
    # 1. Queue Length Over Time
    if sim_metrics.timestamps:
        fig.add_trace(
            go.Scatter(
                x=sim_metrics.timestamps,
                y=sim_metrics.queue_lengths,
                mode='lines',
                name='Simulated Queue',
                line=dict(color='#FF6B6B')
            ),
            row=1, col=1
        )
        
        # Theoretical average line
        if theory_results['avg_queue_length'] != float('inf'):
            fig.add_hline(
                y=theory_results['avg_queue_length'],
                line_dash="dash",
                line_color="#4ECDC4",
                annotation_text=f"Theory: {theory_results['avg_queue_length']:.2f}",
                row=1, col=1
            )
    
    # 2. Utilization Over Time
    if sim_metrics.timestamps:
        fig.add_trace(
            go.Scatter(
                x=sim_metrics.timestamps,
                y=sim_metrics.utilizations,
                mode='lines',
                name='Simulated Utilization',
                line=dict(color='#45B7D1')
            ),
            row=1, col=2
        )
        
        # Theoretical utilization line
        fig.add_hline(
            y=theory_results['utilization'] * 100,
            line_dash="dash",
            line_color="#96CEB4",
            annotation_text=f"Theory: {theory_results['utilization']*100:.1f}%",
            row=1, col=2
        )
    
    # 3. Wait Time Distribution
    if sim_metrics.wait_times:
        fig.add_trace(
            go.Histogram(
                x=sim_metrics.wait_times,
                nbinsx=50,
                name='Wait Times',
                marker_color='#FF6B6B',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Theoretical average line
        if theory_results['avg_wait_time'] != float('inf'):
            fig.add_vline(
                x=theory_results['avg_wait_time'],
                line_dash="dash",
                line_color="#4ECDC4",
                annotation_text=f"Theory Avg: {theory_results['avg_wait_time']:.3f}s",
                row=2, col=1
            )
    
    # 4. System Time Distribution
    if sim_metrics.system_times:
        fig.add_trace(
            go.Histogram(
                x=sim_metrics.system_times,
                nbinsx=50,
                name='System Times',
                marker_color='#45B7D1',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Theoretical average line
        if theory_results['avg_system_time'] != float('inf'):
            fig.add_vline(
                x=theory_results['avg_system_time'],
                line_dash="dash",
                line_color="#96CEB4",
                annotation_text=f"Theory Avg: {theory_results['avg_system_time']:.3f}s",
                row=2, col=2
            )
    
    fig.update_layout(
        height=700,
        title_text="Theory vs Simulation Comparison",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Queue Length", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Utilization (%)", row=1, col=2)
    fig.update_xaxes(title_text="Wait Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="System Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    return fig


def create_capacity_analysis_plot(arrival_rates: list, num_servers: int, 
                                  service_rate: float) -> go.Figure:
    """Create plot showing how metrics change with arrival rate"""
    
    utilizations = []
    wait_times = []
    queue_lengths = []
    prob_waiting = []
    
    for rate in arrival_rates:
        util = QueueingTheory.utilization(rate, service_rate, num_servers)
        utilizations.append(min(util * 100, 100))
        
        if util < 1:
            wt = QueueingTheory.avg_wait_time(rate, service_rate, num_servers)
            ql = QueueingTheory.avg_queue_length(rate, service_rate, num_servers)
            pw = QueueingTheory.erlang_c(rate, service_rate, num_servers)
        else:
            wt = float('inf')
            ql = float('inf')
            pw = 1.0
        
        wait_times.append(min(wt, 100))  # Cap for visualization
        queue_lengths.append(min(ql, 100))
        prob_waiting.append(pw * 100)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Utilization vs Arrival Rate',
            'Avg Wait Time vs Arrival Rate',
            'Avg Queue Length vs Arrival Rate',
            'Probability of Waiting vs Arrival Rate'
        )
    )
    
    fig.add_trace(
        go.Scatter(x=arrival_rates, y=utilizations, mode='lines',
                  name='Utilization', line=dict(color='#FF6B6B', width=3)),
        row=1, col=1
    )
    fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                  annotation_text="80% Target", row=1, col=1)
    
    fig.add_trace(
        go.Scatter(x=arrival_rates, y=wait_times, mode='lines',
                  name='Wait Time', line=dict(color='#45B7D1', width=3)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=arrival_rates, y=queue_lengths, mode='lines',
                  name='Queue Length', line=dict(color='#96CEB4', width=3)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=arrival_rates, y=prob_waiting, mode='lines',
                  name='P(Waiting)', line=dict(color='#DDA0DD', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text=f"Capacity Analysis ({num_servers} servers, service rate={service_rate:.2f}/s)",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Arrival Rate (req/s)", row=1, col=1)
    fig.update_yaxes(title_text="Utilization (%)", row=1, col=1)
    fig.update_xaxes(title_text="Arrival Rate (req/s)", row=1, col=2)
    fig.update_yaxes(title_text="Wait Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Arrival Rate (req/s)", row=2, col=1)
    fig.update_yaxes(title_text="Queue Length", row=2, col=1)
    fig.update_xaxes(title_text="Arrival Rate (req/s)", row=2, col=2)
    fig.update_yaxes(title_text="Probability (%)", row=2, col=2)
    
    return fig


def create_server_scaling_plot(arrival_rate: float, service_rate: float,
                               max_servers: int = 20) -> go.Figure:
    """Create plot showing effect of adding servers"""
    
    server_counts = list(range(1, max_servers + 1))
    utilizations = []
    wait_times = []
    queue_lengths = []
    
    for c in server_counts:
        util = QueueingTheory.utilization(arrival_rate, service_rate, c)
        utilizations.append(min(util * 100, 150))
        
        if util < 1:
            wt = QueueingTheory.avg_wait_time(arrival_rate, service_rate, c)
            ql = QueueingTheory.avg_queue_length(arrival_rate, service_rate, c)
        else:
            wt = 10  # Cap for visualization
            ql = 50
        
        wait_times.append(min(wt, 10))
        queue_lengths.append(min(ql, 50))
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Utilization',
            'Avg Wait Time',
            'Avg Queue Length'
        )
    )
    
    # Mark minimum viable servers
    min_servers = QueueingTheory.required_servers(arrival_rate, service_rate, 0.99)
    recommended = QueueingTheory.required_servers(arrival_rate, service_rate, 0.7)
    
    fig.add_trace(
        go.Bar(x=server_counts, y=utilizations, name='Utilization',
               marker_color=['#FF6B6B' if u > 100 else '#96CEB4' for u in utilizations]),
        row=1, col=1
    )
    fig.add_hline(y=80, line_dash="dash", line_color="orange", row=1, col=1)
    
    fig.add_trace(
        go.Bar(x=server_counts, y=wait_times, name='Wait Time',
               marker_color='#45B7D1'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=server_counts, y=queue_lengths, name='Queue Length',
               marker_color='#DDA0DD'),
        row=1, col=3
    )
    
    # Add annotations for recommendations
    fig.add_vline(x=min_servers, line_dash="dash", line_color="red",
                  annotation_text=f"Min: {min_servers}", row=1, col=1)
    fig.add_vline(x=recommended, line_dash="dash", line_color="green",
                  annotation_text=f"Rec: {recommended}", row=1, col=1)
    
    fig.update_layout(
        height=400,
        title_text=f"Server Scaling Analysis (λ={arrival_rate:.1f}/s, μ={service_rate:.2f}/s)",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="# Servers")
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="Seconds", row=1, col=2)
    fig.update_yaxes(title_text="Requests", row=1, col=3)
    
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(layout="wide", page_title="Capacity Planning Simulator")

st.title("📊 Capacity Planning Simulator")

st.markdown("""
Combine **queueing theory** (Little's Law, M/M/c models) with **discrete event simulation** 
to predict queue delays, utilization, and required capacity.

**Key Concepts**:
- **Little's Law**: L = λW (avg queue = arrival rate × avg wait)
- **Utilization**: ρ = λ / (c × μ) (must be < 1 for stable system)
- **Erlang C**: Probability that arriving request has to wait
- **M/M/c Queue**: Theoretical model for multiple servers with random arrivals
""")

# Sidebar for configuration
st.sidebar.header("⚙️ System Configuration")

st.sidebar.subheader("📥 Arrival Parameters")
arrival_rate_per_min = st.sidebar.slider(
    "Arrival Rate (req/min)", 
    min_value=1.0, max_value=500.0, value=60.0, step=5.0,
    help="Average requests arriving per minute"
)

load_pattern = st.sidebar.selectbox(
    "Load Pattern",
    [lp.value for lp in LoadPattern],
    help="How traffic varies over time"
)

if load_pattern in ["Traffic Spike", "Ramp Up/Down"]:
    spike_multiplier = st.sidebar.slider(
        "Spike/Peak Multiplier", 
        min_value=1.5, max_value=10.0, value=3.0, step=0.5
    )
else:
    spike_multiplier = 3.0

st.sidebar.subheader("🖥️ Server Parameters")
num_servers = st.sidebar.slider(
    "Number of Servers", 
    min_value=1, max_value=50, value=5,
    help="Concurrent service capacity"
)

avg_service_time = st.sidebar.slider(
    "Avg Service Time (s)", 
    min_value=0.01, max_value=5.0, value=0.5, step=0.05,
    help="Time your service takes (excluding dependencies)"
)

st.sidebar.subheader("🔗 Dependency Parameters")
num_dependencies = st.sidebar.slider(
    "Number of Dependencies", 
    min_value=0, max_value=10, value=2,
    help="External calls per request"
)

avg_dependency_latency = st.sidebar.slider(
    "Avg Dependency Latency (s)", 
    min_value=0.0, max_value=2.0, value=0.1, step=0.02,
    help="Latency per dependency call"
)

st.sidebar.subheader("⏱️ Simulation")
duration_minutes = st.sidebar.slider(
    "Duration (minutes)", 
    min_value=1.0, max_value=30.0, value=5.0, step=1.0
)

# Advanced options
with st.sidebar.expander("🔧 Advanced Options"):
    max_queue_size = st.number_input(
        "Max Queue Size (0=unlimited)", 
        min_value=0, max_value=1000, value=0
    )
    max_queue = None if max_queue_size == 0 else max_queue_size
    
    enable_retries = st.checkbox("Enable Retries", value=True)
    if enable_retries:
        max_retries = st.slider("Max Retries", 1, 5, 3)
        enable_backoff = st.checkbox("Enable Backoff", value=True)
    else:
        max_retries = 0
        enable_backoff = False
    
    dep_failure_rate = st.slider(
        "Dependency Failure Rate", 
        min_value=0.0, max_value=0.5, value=0.01, step=0.01
    )
    
    enable_circuit_breaker = st.checkbox("Enable Circuit Breaker", value=False)

# ============================================================================
# THEORETICAL CALCULATIONS
# ============================================================================

st.header("📐 Theoretical Analysis (M/M/c Queue)")

# Calculate total service time including dependencies
total_service_time = avg_service_time + (num_dependencies * avg_dependency_latency)
service_rate = 1.0 / total_service_time  # requests per second
arrival_rate_per_sec = arrival_rate_per_min / 60.0

# Calculate theoretical metrics
theory_results = {
    'utilization': QueueingTheory.utilization(arrival_rate_per_sec, service_rate, num_servers),
    'erlang_c': QueueingTheory.erlang_c(arrival_rate_per_sec, service_rate, num_servers),
    'avg_queue_length': QueueingTheory.avg_queue_length(arrival_rate_per_sec, service_rate, num_servers),
    'avg_wait_time': QueueingTheory.avg_wait_time(arrival_rate_per_sec, service_rate, num_servers),
    'avg_system_time': QueueingTheory.avg_system_time(arrival_rate_per_sec, service_rate, num_servers),
    'p95_wait_time': QueueingTheory.percentile_wait_time(arrival_rate_per_sec, service_rate, num_servers, 0.95),
    'required_servers_80': QueueingTheory.required_servers(arrival_rate_per_sec, service_rate, 0.8),
    'required_servers_70': QueueingTheory.required_servers(arrival_rate_per_sec, service_rate, 0.7),
    'max_arrival_rate': QueueingTheory.max_arrival_rate(service_rate, num_servers, 0.8),
}

# Display theoretical results
col1, col2, col3, col4 = st.columns(4)

with col1:
    util_color = "🟢" if theory_results['utilization'] < 0.7 else "🟡" if theory_results['utilization'] < 0.9 else "🔴"
    st.metric(
        f"{util_color} Utilization (ρ)", 
        f"{theory_results['utilization']*100:.1f}%",
        help="ρ = λ/(c×μ). Should be < 80% for stable performance"
    )

with col2:
    st.metric(
        "📊 Avg Queue Length (Lq)", 
        f"{theory_results['avg_queue_length']:.2f}" if theory_results['avg_queue_length'] != float('inf') else "∞",
        help="Average number of requests waiting"
    )

with col3:
    st.metric(
        "⏱️ Avg Wait Time (Wq)", 
        f"{theory_results['avg_wait_time']*1000:.1f} ms" if theory_results['avg_wait_time'] != float('inf') else "∞",
        help="Average time spent waiting in queue"
    )

with col4:
    st.metric(
        "🎯 P(Waiting)", 
        f"{theory_results['erlang_c']*100:.1f}%",
        help="Probability that a request has to wait (Erlang C)"
    )

# Second row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🔄 Avg System Time (W)", 
        f"{theory_results['avg_system_time']*1000:.1f} ms" if theory_results['avg_system_time'] != float('inf') else "∞",
        help="Total time in system (wait + service)"
    )

with col2:
    st.metric(
        "📈 P95 Wait Time", 
        f"{theory_results['p95_wait_time']*1000:.1f} ms" if theory_results['p95_wait_time'] != float('inf') else "∞",
        help="95th percentile wait time"
    )

with col3:
    st.metric(
        "🖥️ Servers for 70% Util", 
        theory_results['required_servers_70'],
        help="Recommended servers for healthy utilization"
    )

with col4:
    st.metric(
        "📊 Max Rate at 80% Util", 
        f"{theory_results['max_arrival_rate']*60:.0f}/min",
        help="Maximum sustainable arrival rate"
    )

# Little's Law demonstration
st.subheader("📏 Little's Law Verification")

littles_law_l = arrival_rate_per_sec * theory_results['avg_system_time'] if theory_results['avg_system_time'] != float('inf') else float('inf')
littles_law_display = f"{littles_law_l:.2f}" if littles_law_l != float('inf') else "∞"

col1, col2, col3 = st.columns(3)
with col1:
    st.latex(r"L = \lambda \times W")
with col2:
    st.markdown(f"**L** (avg in system) = {arrival_rate_per_sec:.2f} × {theory_results['avg_system_time']:.4f} = **{littles_law_display}**")
with col3:
    if theory_results['utilization'] < 1:
        direct_l = QueueingTheory.avg_requests_in_system(arrival_rate_per_sec, service_rate, num_servers)
        st.markdown(f"Direct calculation: **{direct_l:.2f}** ✓")
    else:
        st.markdown("System unstable (ρ ≥ 1)")

# Stability warning
if theory_results['utilization'] >= 1:
    st.error(f"""
    ⚠️ **System Unstable!**
    
    Utilization ρ = {theory_results['utilization']:.2f} ≥ 1.0
    
    The arrival rate exceeds service capacity. Queue will grow unbounded.
    
    **Solutions**:
    - Increase servers to at least **{theory_results['required_servers_80']}**
    - Reduce arrival rate to below **{theory_results['max_arrival_rate']*60:.0f}/min**
    - Reduce service time (optimize code, reduce dependencies)
    """)
elif theory_results['utilization'] >= 0.9:
    st.warning(f"""
    ⚠️ **High Utilization Warning**
    
    Utilization ρ = {theory_results['utilization']*100:.1f}% is high.
    
    System will experience:
    - Long queue times
    - High latency variance  
    - Vulnerability to traffic spikes
    
    **Recommendation**: Add servers to reach 70-80% utilization.
    """)

# ============================================================================
# CAPACITY ANALYSIS PLOTS
# ============================================================================

st.header("📈 Capacity Analysis")

tab1, tab2 = st.tabs(["📊 Arrival Rate Impact", "🖥️ Server Scaling"])

with tab1:
    st.markdown("How do metrics change as arrival rate increases?")
    
    # Generate range of arrival rates
    max_rate = theory_results['max_arrival_rate'] * 1.5  # Go beyond max
    arrival_rates = np.linspace(0.1, max_rate, 100)
    
    fig = create_capacity_analysis_plot(arrival_rates, num_servers, service_rate)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    📌 **Key Points**:
    - System becomes unstable when utilization hits 100% (ρ = 1)
    - Current max sustainable rate: **{theory_results['max_arrival_rate']*60:.0f} req/min** at 80% utilization
    - Wait times grow exponentially as utilization approaches 100%
    """)

with tab2:
    st.markdown("How many servers do you need?")
    
    fig = create_server_scaling_plot(arrival_rate_per_sec, service_rate, 20)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    📌 **Recommendations for {arrival_rate_per_min:.0f} req/min**:
    - **Minimum servers**: {theory_results['required_servers_80']} (for < 100% utilization)
    - **Recommended**: {theory_results['required_servers_70']} (for 70% utilization)
    - **Current**: {num_servers} servers ({theory_results['utilization']*100:.1f}% utilization)
    """)

# ============================================================================
# SIMULATION
# ============================================================================

st.header("🔬 Simulation")

if st.button("🚀 Run Simulation", type="primary"):
    
    # Create configuration
    config = SimulationConfig(
        base_arrival_rate=arrival_rate_per_min,
        avg_service_time=avg_service_time,
        num_dependencies=num_dependencies,
        avg_dependency_latency=avg_dependency_latency,
        num_servers=num_servers,
        max_queue_size=max_queue,
        enable_retries=enable_retries,
        max_retries=max_retries if enable_retries else 0,
        retry_backoff_enabled=enable_backoff if enable_retries else False,
        dependency_failure_rate=dep_failure_rate,
        enable_circuit_breaker=enable_circuit_breaker,
        load_pattern=LoadPattern(load_pattern),
        spike_multiplier=spike_multiplier,
        duration_minutes=duration_minutes,
    )
    
    with st.spinner("Running simulation..."):
        simulator = CapacitySimulator(config)
        metrics = simulator.run()
    
    st.success("Simulation complete!")
    
    # Display simulation metrics
    st.subheader("📊 Simulation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Arrivals", metrics.arrivals)
    with col2:
        st.metric("Completions", metrics.completions)
    with col3:
        st.metric("Rejections", metrics.rejections)
    with col4:
        success_rate = (metrics.completions / metrics.arrivals * 100) if metrics.arrivals > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Retries", metrics.retries)
    with col2:
        st.metric("Timeouts", metrics.timeouts)
    with col3:
        st.metric("Dep Failures", metrics.dependency_failures)
    with col4:
        st.metric("Circuit Breaker", metrics.circuit_breaker_rejections)
    
    # Comparison with theory
    st.subheader("🔄 Theory vs Simulation Comparison")
    
    if metrics.wait_times and metrics.system_times:
        sim_avg_wait = np.mean(metrics.wait_times)
        sim_avg_system = np.mean(metrics.system_times)
        sim_p95_wait = np.percentile(metrics.wait_times, 95)
        sim_avg_queue = np.mean(metrics.queue_lengths)
        sim_avg_util = np.mean(metrics.utilizations)
        
        comparison_data = {
            "Metric": ["Avg Wait Time (ms)", "Avg System Time (ms)", "P95 Wait Time (ms)", 
                      "Avg Queue Length", "Avg Utilization (%)"],
            "Theory": [
                f"{theory_results['avg_wait_time']*1000:.2f}" if theory_results['avg_wait_time'] != float('inf') else "∞",
                f"{theory_results['avg_system_time']*1000:.2f}" if theory_results['avg_system_time'] != float('inf') else "∞",
                f"{theory_results['p95_wait_time']*1000:.2f}" if theory_results['p95_wait_time'] != float('inf') else "∞",
                f"{theory_results['avg_queue_length']:.2f}" if theory_results['avg_queue_length'] != float('inf') else "∞",
                f"{theory_results['utilization']*100:.1f}"
            ],
            "Simulation": [
                f"{sim_avg_wait*1000:.2f}",
                f"{sim_avg_system*1000:.2f}",
                f"{sim_p95_wait*1000:.2f}",
                f"{sim_avg_queue:.2f}",
                f"{sim_avg_util:.1f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Plot comparison
        fig = create_theory_vs_simulation_plot(theory_results, metrics, config)
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis
    st.subheader("🔍 Analysis")
    
    if metrics.wait_times:
        sim_avg_wait = np.mean(metrics.wait_times)
        
        if theory_results['utilization'] >= 1:
            st.error("""
            System was **unstable** during simulation. Utilization ≥ 100%.
            
            This means arrivals exceed capacity and queue grew throughout the simulation.
            """)
        elif sim_avg_wait > theory_results['avg_wait_time'] * 2 and theory_results['avg_wait_time'] != float('inf'):
            st.warning(f"""
            ⚠️ Simulation wait times ({sim_avg_wait*1000:.1f}ms) significantly higher than theory ({theory_results['avg_wait_time']*1000:.1f}ms).
            
            Possible causes:
            - Dependency failures causing retries
            - Non-exponential service times (variance matters)
            - Traffic pattern effects (spikes, bursts)
            - Circuit breaker activations
            """)
        else:
            st.success("""
            ✅ Simulation results align with theoretical predictions.
            
            The M/M/c model provides a reasonable approximation for this workload.
            """)

# Educational sections
with st.expander("📚 Queueing Theory Fundamentals"):
    st.markdown("""
    ### Little's Law (1961)
    
    **L = λW**
    
    Where:
    - L = Average number of items in system
    - λ = Arrival rate
    - W = Average time in system
    
    **Powerful because**: Holds for ANY queueing system in steady state, regardless of arrival distribution or service distribution.
    
    ### M/M/c Queue Model
    
    **Assumptions**:
    - **M**: Markovian (Poisson) arrivals
    - **M**: Markovian (Exponential) service times
    - **c**: Number of parallel servers
    
    **Key Formulas**:
    
    **Utilization**: ρ = λ / (c × μ)
    
    **Erlang C** (probability of waiting):
    
    For stable queue (ρ < 1), gives probability that arriving request must wait.
    
    **Average Wait Time**: Wq = P(wait) × 1/(cμ - λ)
    
    ### When Theory Breaks Down
    
    Real systems differ from M/M/c because:
    - Service times aren't exponential (often have longer tails)
    - Arrivals may be correlated or bursty
    - Servers may have different speeds
    - Dependencies add complexity
    - Retries create positive feedback loops
    
    **That's why we simulate!** Theory gives baseline, simulation validates.
    """)

with st.expander("🎯 Capacity Planning Best Practices"):
    st.markdown("""
    ### Target Utilization
    
    | Utilization | Status | Recommendation |
    |-------------|--------|----------------|
    | < 50% | Over-provisioned | Consider scaling down |
    | 50-70% | Healthy | Good headroom for spikes |
    | 70-80% | Optimal | Efficient but watch closely |
    | 80-90% | Stressed | Plan to scale soon |
    | > 90% | Critical | Scale immediately |
    
    ### The Utilization Trap
    
    **Why 70-80% and not 95%?**
    
    Queue wait times grow **exponentially** as utilization approaches 100%:
    - At 50% util: ~1x service time wait
    - At 80% util: ~4x service time wait
    - At 90% util: ~9x service time wait
    - At 95% util: ~19x service time wait
    
    Small traffic spikes at high utilization cause massive latency increases.
    
    ### Capacity Planning Process
    
    1. **Measure current load**: Arrival rate, service time, dependency latency
    2. **Calculate utilization**: ρ = λ / (c × μ)
    3. **Project growth**: Expected traffic increase
    4. **Determine target**: Usually 70% utilization at peak
    5. **Calculate required capacity**: c = λ / (ρ_target × μ)
    6. **Validate with simulation**: Test edge cases
    7. **Plan scaling triggers**: Automated scaling rules
    
    ### Dependencies Matter
    
    Your service time = your code + all dependencies
    
    If you have 2 dependencies at 100ms each:
    - Your code: 50ms
    - **Total service time: 250ms**
    - This 5x increase dramatically affects capacity!
    
    **Lesson**: Optimize dependencies, not just your code.
    """)

with st.expander("🔬 Experiment Ideas"):
    st.markdown("""
    ### Experiment 1: Find the Breaking Point
    1. Start with low arrival rate
    2. Gradually increase until utilization > 100%
    3. Observe how queue length explodes
    4. Note the "cliff" in performance
    
    ### Experiment 2: Traffic Spike Impact
    1. Set load pattern to "Traffic Spike"
    2. Run with 3x spike multiplier
    3. Observe queue buildup during spike
    4. Watch recovery time after spike
    5. Compare with different server counts
    
    ### Experiment 3: Dependency Failure Cascade
    1. Enable retries
    2. Set dependency failure rate to 10%
    3. Observe retry amplification
    4. Enable circuit breaker
    5. See how circuit breaker limits damage
    
    ### Experiment 4: Theory vs Reality
    1. Run with constant load (most like M/M/c)
    2. Compare theory vs simulation
    3. Switch to bursty load
    4. See how theory underestimates wait times
    
    ### Experiment 5: Optimal Server Count
    1. Use server scaling plot
    2. Find minimum servers for stability
    3. Find recommended servers for 70% util
    4. Simulate both to compare latency
    """)
